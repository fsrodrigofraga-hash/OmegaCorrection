# ============================================================
# OmegaGW_MG
# Population-level energy-density spectrum (Omega_GW) builder with
# optional multiplicative correction factors.
#
# Backend: popstock (population Monte Carlo) + bilby (waveforms).
#
# The module provides:
#   - builders for the reusable population/waveform context,
#   - helpers to assemble a per-event dataset from population samples,
#   - a (optionally approximated) cached waveform-power routine,
#   - a small set of multiplicative correction terms combined by an
#     orchestrator function.
#
# Public entry points are kept stable; internal helpers (prefixed with
# an underscore) are implementation details and may change.
# ============================================================

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional, Union

from multiprocessing import Pool, cpu_count

import bilby
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, c, M_sun
from scipy.signal import savgol_filter

from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from popstock.PopulationOmegaGW import PopulationOmegaGW


# ============================================================
# Type aliases
# ============================================================
EventWeightFn = Callable[[Dict[str, np.ndarray]], np.ndarray]

# Coefficient: scalar (broadcast to all events) or callable(dataset)->scalar/array.
AlphaLike = Union[float, Callable[[Dict[str, np.ndarray]], Union[float, np.ndarray]]]


def resolve_alpha(alpha_ppE: AlphaLike, dataset: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize the coefficient into a per-event array of length N_sys.

    Accepts a scalar (broadcast to all events) or a callable(dataset) that
    returns a scalar or an array of length N_sys.
    """
    if "mass_1" not in dataset:
        raise KeyError("dataset must contain key 'mass_1' to determine N_sys")
    N = len(dataset["mass_1"])

    if callable(alpha_ppE):
        a = alpha_ppE(dataset)
    else:
        a = alpha_ppE

    a = np.asarray(a, dtype=float)

    if a.ndim == 0:
        return np.full(N, float(a), dtype=float)
    if a.shape == (N,):
        return a.astype(float, copy=False)

    raise ValueError(f"alpha_ppE must be float or callable returning shape (N_sys,), got {a.shape}")



# ============================================================
# Configuration dataclasses
# ============================================================
@dataclass(frozen=True)
class WaveformConfig:
    """Waveform generator configuration (frequency-domain compact-binary waveform)."""
    duration: float = 4
    sampling_frequency: float = 4096
    binary_type: str = "BBH"  # "BBH", "BNS", "BHNS", "NSBH"
    waveform_approximant: str = "IMRPhenomD"
    reference_frequency: float = 25.0
    minimum_frequency: float = 10.0
    maximum_frequency: Optional[float] = None  # passed to waveform_arguments when supported

    # Optional behavior flags (defaults preserve the standard path).
    inspiral_only: bool = False
    disable_inspiral_cutoff: bool = False


@dataclass(frozen=True)
class FreqGridConfig:
    """Reduced frequency grid configuration (log-spaced)."""
    fmin: float = 10.0
    fmax: float = 4096.0
    N_freq_eff: int = 2000


@dataclass(frozen=True)
class CorrectionContext:
    """
    Reusable context holding fixed objects/settings:
    - population models (mass + redshift)
    - waveform config
    - reduced frequency grid and mapping indices into the full waveform frequency array
    """
    models: Dict[str, Any]
    wf_cfg: WaveformConfig
    fgrid_cfg: FreqGridConfig
    frequencies: np.ndarray
    idx_freq: np.ndarray


# ============================================================
# Builders: waveform generator, frequency grid, population models
# ============================================================
def build_models(z_max: float = 10.0) -> Dict[str, Any]:
    """Build the default population models (mass + redshift).

    Dict keys follow the names expected by the population backend.
    """
    return {
        "mass": SinglePeakSmoothedMassDistribution(),
        "redshift": MadauDickinsonRedshift(z_max=z_max),
    }



# ============================================================
# Compact-binary type helpers (BBH / BNS / BHNS / NSBH)
# ============================================================
def _normalize_binary_type(binary_type: Any) -> str:
    """Normalize user input into one of: BBH, BNS, BHNS, NSBH."""
    if binary_type is None:
        return "BBH"
    s = str(binary_type).strip().upper()
    s = s.replace("-", "").replace("_", "")

    if s in ("BBH", "BHBH", "BINARYBLACKHOLE"):
        return "BBH"
    if s in ("BNS", "NSNS", "BINARYNEUTRONSTAR"):
        return "BNS"
    if s in ("BHNS", "NSBH", "NEUTRONSTARBLACKHOLE", "BLACKHOLENEUTRONSTAR"):
        # preserve explicit ordering if provided
        return "BHNS" if s == "BHNS" else "NSBH"
    return s


def _resolve_bilby_source_and_conversion(binary_type: Any):
    """Return (frequency_domain_source_model, parameter_conversion, normalized_type)."""
    bt = _normalize_binary_type(binary_type)

    # BBH
    if bt == "BBH":
        return (
            bilby.gw.source.lal_binary_black_hole,
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            bt,
        )

    # BNS
    if bt == "BNS":
        src = getattr(bilby.gw.source, "lal_binary_neutron_star", None)
        conv = getattr(bilby.gw.conversion, "convert_to_lal_binary_neutron_star_parameters", None)
        if (src is None) or (conv is None):
            warnings.warn(
                "bilby does not expose BNS waveform helpers "
                "(lal_binary_neutron_star / convert_to_lal_binary_neutron_star_parameters) "
                "in this installation. Falling back to BBH source model/conversion.",
                RuntimeWarning,
            )
            return (
                bilby.gw.source.lal_binary_black_hole,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                bt,
            )
        return (src, conv, bt)

    # NSBH / BHNS (mixed)
    if bt in ("NSBH", "BHNS"):
        # bilby exposes BBH and BNS source models; BHNS/NSBH systems are typically handled
        # by using the BNS source model with one object's tidal deformability set to 0.
        # (i.e., set lambda_1/lambda_2 appropriately in the dataset).
        src = getattr(bilby.gw.source, "lal_binary_neutron_star", None)
        conv = getattr(bilby.gw.conversion, "convert_to_lal_binary_neutron_star_parameters", None)

        if (src is None) or (conv is None):
            warnings.warn(
                "bilby does not expose BNS waveform helpers "
                "(lal_binary_neutron_star / convert_to_lal_binary_neutron_star_parameters) "
                "in this installation. Falling back to BBH source model/conversion (no tidal effects).",
                RuntimeWarning,
            )
            return (
                bilby.gw.source.lal_binary_black_hole,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                bt,
            )

        return (src, conv, bt)

    raise ValueError(f"Unsupported binary_type={binary_type!r}. Use one of: 'BBH', 'BNS', 'BHNS', 'NSBH'.")


def build_waveform_generator(cfg: WaveformConfig) -> bilby.gw.WaveformGenerator:
    """Create a Bilby waveform generator for frequency-domain compact-binary waveforms."""
    approximant = cfg.waveform_approximant

    wargs = {
        "waveform_approximant": approximant,
        "reference_frequency": cfg.reference_frequency,
        "minimum_frequency": cfg.minimum_frequency,
    }
    if cfg.maximum_frequency is not None:
        # Some bilby/LAL paths accept this; if not, it will be ignored downstream.
        wargs["maximum_frequency"] = float(cfg.maximum_frequency)

    src_model, conv, _ = _resolve_bilby_source_and_conversion(getattr(cfg, "binary_type", "BBH"))

    return bilby.gw.WaveformGenerator(
        duration=cfg.duration,
        sampling_frequency=cfg.sampling_frequency,
        frequency_domain_source_model=src_model,
        parameter_conversion=conv,
        waveform_arguments=wargs,
    )



def build_frequency_grid(full_frequencies: np.ndarray, cfg: FreqGridConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a reduced log-spaced frequency grid and indices that map each reduced frequency
    to the closest (or next) entry in the full frequency array.
    """
    fmax_eff = min(float(cfg.fmax), float(full_frequencies[-1]))
    if cfg.fmax > full_frequencies[-1]:
        warnings.warn(
            f"Requested fmax={cfg.fmax} Hz, but waveform Nyquist is {full_frequencies[-1]:.1f} Hz. "
            f"Clipping fmax to {fmax_eff:.1f} Hz. "
            "If you truly want 4096 Hz, set sampling_frequency >= 8192.",
            RuntimeWarning,
        )
    freqs = np.logspace(np.log10(cfg.fmin), np.log10(fmax_eff), int(cfg.N_freq_eff))
    idx = np.searchsorted(full_frequencies, freqs)
    idx = np.clip(idx, 0, len(full_frequencies) - 1)
    return freqs, idx


def prepare_context(
    wf_cfg: WaveformConfig = WaveformConfig(),
    fgrid_cfg: FreqGridConfig = FreqGridConfig(),
    z_max_models: float = 10.0,
) -> CorrectionContext:
    """
    Prepare reusable context once:
    - Build models
    - Build waveform generator (only to get its full frequency array)
    - Build reduced frequency grid + mapping indices
    """
    models = build_models(z_max=z_max_models)

    wg = build_waveform_generator(wf_cfg)
    full_f = wg.frequency_array

    frequencies, idx = build_frequency_grid(full_f, fgrid_cfg)

    return CorrectionContext(
        models=models,
        wf_cfg=wf_cfg,
        fgrid_cfg=fgrid_cfg,
        frequencies=frequencies,
        idx_freq=idx,
    )


# ============================================================
# Popstock samples -> dataset
# ============================================================
def _as_dict_of_arrays(obj: Any) -> Dict[str, np.ndarray]:
    """Convert dict/structured array/Mapping-like into {key: np.ndarray}."""
    if obj is None:
        raise ValueError("Cannot convert None to a sample dict.")

    if isinstance(obj, dict):
        return {k: np.asarray(v) for k, v in obj.items()}

    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        return {name: np.asarray(obj[name]) for name in obj.dtype.names}

    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        return {k: np.asarray(obj[k]) for k in obj.keys()}

    raise TypeError(f"Unsupported sample container type: {type(obj)}")


def _extract_popstock_samples(pop: PopulationOmegaGW) -> Dict[str, np.ndarray]:
    """
    Try hard to extract the proposal samples from a PopulationOmegaGW instance.
    """
    candidate_attrs = [
        "proposal_samples",
        "proposal_samples_dict",
        "proposal",
        "samples",
        "sample_dict",
        "_proposal_samples",
        "_samples",
    ]

    for attr in candidate_attrs:
        if hasattr(pop, attr):
            val = getattr(pop, attr)
            if val is not None:
                try:
                    return _as_dict_of_arrays(val)
                except Exception:
                    pass

    for k, v in getattr(pop, "__dict__", {}).items():
        if "sample" in k.lower() and v is not None:
            try:
                return _as_dict_of_arrays(v)
            except Exception:
                continue

    raise AttributeError(
        "Could not find proposal samples inside PopulationOmegaGW.\n"
        "Tip: print([k for k in pop.__dict__.keys() if 'sample' in k.lower()]) "
        "and adapt _extract_popstock_samples(...) to your popstock installation."
    )


def _pick_key(d: Dict[str, np.ndarray], *names: str) -> Optional[str]:
    """Return the first key from ``names`` that is present in the dict."""
    for n in names:
        if n in d:
            return n
    return None


def dataset_from_popstock_samples(
    pop: PopulationOmegaGW,
    *,
    set_extrinsics: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build dataset dict using mass + redshift drawn by popstock (proposal samples in 'pop').
    """
    s = _extract_popstock_samples(pop)

    k_m1 = _pick_key(s, "mass_1", "m1", "m1_source", "mass_1_source")
    k_m2 = _pick_key(s, "mass_2", "m2", "m2_source", "mass_2_source")
    k_q  = _pick_key(s, "mass_ratio", "q", "massratio")
    k_z  = _pick_key(s, "redshift", "z", "z_source")

    if k_m1 is None:
        raise KeyError(f"Could not find mass_1 key. Available: {list(s.keys())}")
    if k_z is None:
        raise KeyError(f"Could not find redshift key. Available: {list(s.keys())}")

    m1 = np.asarray(s[k_m1], dtype=float)
    z  = np.asarray(s[k_z], dtype=float)

    if k_m2 is not None:
        m2 = np.asarray(s[k_m2], dtype=float)
        q = m2 / m1
    elif k_q is not None:
        q = np.asarray(s[k_q], dtype=float)
        m2 = m1 * q
    else:
        raise KeyError(f"Need mass_2 or mass_ratio. Available: {list(s.keys())}")

    N = len(m1)
    if len(m2) != N or len(z) != N:
        raise ValueError("Sample arrays have inconsistent lengths.")

    dataset: Dict[str, np.ndarray] = {
        "mass_1": m1,
        "mass_2": m2,
        "mass_ratio": q,
        "redshift": z,
    }

    if set_extrinsics:
        dataset.update({
            "a_1": np.zeros(N),
            "a_2": np.zeros(N),
            "tilt_1": np.zeros(N),
            "tilt_2": np.zeros(N),
            "phi_12": np.zeros(N),
            "phi_jl": np.zeros(N),
            "lambda_1": np.zeros(N),
            "lambda_2": np.zeros(N),
            "luminosity_distance": cosmo.luminosity_distance(z).value,  # Mpc
            "theta_jn": np.zeros(N),
            "phase": np.zeros(N),
            "geocent_time": np.zeros(N),
        })

    return dataset


def draw_pop_and_dataset_from_popstock(
    *,
    models: Dict[str, Any],
    Lambda: Dict[str, float],
    frequency_array: np.ndarray,
    N_sys: int,
    set_extrinsics: bool = True,
) -> Tuple[PopulationOmegaGW, Dict[str, np.ndarray]]:
    """Draw proposal samples for the given hyperparameters and build the
    matching per-event dataset."""
    pop = PopulationOmegaGW(models=models, frequency_array=frequency_array)
    pop.draw_and_set_proposal_samples(Lambda, N_proposal_samples=int(N_sys))
    dataset = dataset_from_popstock_samples(pop, set_extrinsics=set_extrinsics)
    return pop, dataset


# ============================================================
# Population weights
# ============================================================
def calculate_probabilities(
    dataset: Dict[str, np.ndarray],
    Lambda: Dict[str, float],
    models: Dict[str, Any],
) -> np.ndarray:
    """Evaluate the population probability weights for the dataset under the
    given hyperparameters."""
    pop_tmp = PopulationOmegaGW(models=models)
    return pop_tmp.calculate_probabilities(dataset, Lambda)


# ============================================================
# Inspiral cutoff helpers
# ============================================================
def f_isco_source_hz(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-event characteristic frequency in the source frame."""
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    Mtot_kg = (m1 + m2) * M_sun.value
    return c.value**3 / ((6.0**1.5) * np.pi * G.value * Mtot_kg)


def robust_inspiral_end_frequency_obs(
    dataset: Dict[str, np.ndarray],
    f_peak_obs: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the per-event characteristic frequency with an optional
    secondary estimate into an observed-frame end-of-band value.

    Returns (f_end_obs, f_char_obs, f_char_src).
    """
    f_isco_src = f_isco_source_hz(dataset)
    z = np.asarray(dataset["redshift"], dtype=float)
    f_isco_obs = f_isco_src / (1.0 + z)

    if f_peak_obs is None:
        return f_isco_obs.copy(), f_isco_obs, f_isco_src

    f_peak_obs = np.asarray(f_peak_obs, dtype=float)
    f_insp_end_obs = f_isco_obs.copy()

    good = np.isfinite(f_peak_obs) & (f_peak_obs > 0)
    f_insp_end_obs[good] = np.minimum(f_isco_obs[good], f_peak_obs[good])

    return f_insp_end_obs, f_isco_obs, f_isco_src


# ============================================================
# Waveform |h(f)|^2 cache (EXACT worker)
# ============================================================
_WORK_DATASET = None
_WORK_IDX = None
_WORK_WFGEN = None
_WORK_FULL_F = None
_WORK_FMIN = None
_WORK_INSPIRAL_ONLY = False

_WF_PARAM_KEYS = (
    "mass_1", "mass_2",
    "luminosity_distance",
    "theta_jn", "phase",
    "a_1", "a_2",
    "tilt_1", "tilt_2",
    "phi_12", "phi_jl",
    "lambda_1", "lambda_2",
    "geocent_time",
)

def _worker_init(dataset: Dict[str, np.ndarray], idx: np.ndarray, wf_cfg: WaveformConfig) -> None:
    """Initialize per-process global state for the parallel waveform workers."""
    global _WORK_DATASET, _WORK_IDX, _WORK_WFGEN, _WORK_FULL_F, _WORK_FMIN, _WORK_INSPIRAL_ONLY
    _WORK_DATASET = dataset
    _WORK_IDX = idx
    _WORK_WFGEN = build_waveform_generator(wf_cfg)
    _WORK_FULL_F = _WORK_WFGEN.frequency_array
    _WORK_FMIN = float(wf_cfg.minimum_frequency)
    _WORK_INSPIRAL_ONLY = bool(getattr(wf_cfg, "inspiral_only", False))


def _compute_h2_and_fpeak_worker(i: int) -> Tuple[np.ndarray, float]:
    """Per-event worker: return the binned waveform power and a
    characteristic frequency (NaN when not computed)."""
    ds = _WORK_DATASET
    idx = _WORK_IDX
    wg = _WORK_WFGEN
    full_f = _WORK_FULL_F
    fmin = _WORK_FMIN

    params = {k: float(ds[k][i]) for k in _WF_PARAM_KEYS}
    pol = wg.frequency_domain_strain(params)

    h2_full = np.abs(pol["plus"])**2 + np.abs(pol["cross"])**2

    # Optional fast path: skip the characteristic-frequency estimate.
    if _WORK_INSPIRAL_ONLY:
        return h2_full[idx], float("nan")

    # Characteristic-frequency estimate from a weighted power score.
    score = h2_full * np.power(full_f, 7.0 / 3.0, where=(full_f > 0), out=np.zeros_like(full_f))

    band = (
        (full_f >= fmin) &
        (full_f > 1.05 * fmin) &
        np.isfinite(score)
    )

    if not np.any(band):
        f_peak = np.nan
    else:
        j = int(np.argmax(score[band]))
        f_peak = float(full_f[band][j])

    return h2_full[idx], f_peak


def _compute_exact_h2_cache_and_fpeak_parallel(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    nproc: Optional[int],
    chunksize: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact (one waveform per event) parallel computation of the
    waveform-power cache and per-event characteristic frequencies."""
    N_sys = len(dataset["mass_1"])
    nproc = cpu_count() if nproc is None else int(nproc)

    with Pool(
        processes=nproc,
        initializer=_worker_init,
        initargs=(dataset, idx, wf_cfg),
    ) as pool:
        out = pool.map(_compute_h2_and_fpeak_worker, range(N_sys), chunksize=int(chunksize))

    h2_list, fpeak_list = zip(*out)
    return np.asarray(h2_list), np.asarray(fpeak_list, dtype=float)


# ============================================================
# Approximate (binned) waveform-power cache
# ============================================================
def _chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Standard chirp-mass combination."""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def _safe_interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Simple robust linear interpolation with endpoint clipping (x must be increasing)."""
    return np.interp(xq, x, y, left=y[0], right=y[-1])

def _choose_bins_counts(max_bins: int) -> Tuple[int, int]:
    """Split a total bin budget into a 2D (rows, cols) grid count."""
    nm = int(np.sqrt(max_bins))
    nq = max(2, int(max_bins / max(2, nm)))
    nm = max(2, nm)
    return nm, nq

def _assign_quantile_bins(x: np.ndarray, nbin: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (bin_id, edges). bin_id in [0, nbin-1].
    Uses quantile edges to balance counts.
    """
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0.0, 1.0, int(nbin) + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.zeros_like(x, dtype=int), edges
    internal = edges[1:-1]
    bid = np.digitize(x, internal, right=False)
    return bid.astype(int), edges

def _compute_fast_binned_h2_cache_and_fpeak(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    frequencies: np.ndarray,
    nproc: Optional[int],
    chunksize: int,
    max_bins: int = 300,
    use_frequency_warp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate the waveform-power cache and characteristic frequencies
    using a binned, representative-event scheme that reduces the number of
    full waveform evaluations.
    """
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    q  = np.asarray(dataset["mass_ratio"], dtype=float)
    z  = np.asarray(dataset["redshift"], dtype=float)
    dl = np.asarray(dataset["luminosity_distance"], dtype=float)

    Mc = _chirp_mass(m1, m2)
    Mcz = Mc * (1.0 + z)
    Mtotz = (m1 + m2) * (1.0 + z)

    nm, nq = _choose_bins_counts(int(max_bins))

    b_m, _ = _assign_quantile_bins(np.log10(Mcz + 1e-30), nm)
    b_q, _ = _assign_quantile_bins(q, nq)

    gid = b_m * nq + b_q
    uniq = np.unique(gid)

    reps = []
    groups = []
    for g in uniq:
        idxs = np.where(gid == g)[0]
        if idxs.size == 0:
            continue
        reps.append(int(idxs[idxs.size // 2]))
        groups.append(idxs)

    reps = np.asarray(reps, dtype=int)
    if reps.size == 0:
        raise RuntimeError("No groups built for fast_binning (unexpected).")

    nproc_eff = cpu_count() if nproc is None else int(nproc)

    with Pool(
        processes=nproc_eff,
        initializer=_worker_init,
        initargs=(dataset, idx, wf_cfg),
    ) as pool:
        out = pool.map(_compute_h2_and_fpeak_worker, reps.tolist(), chunksize=max(1, int(chunksize)))

    rep_h2_list, rep_fpeak_list = zip(*out)
    rep_h2 = np.asarray(rep_h2_list)               # (n_groups, Nf)
    rep_fpeak = np.asarray(rep_fpeak_list, float)  # (n_groups,)

    N = len(m1)
    Nf = len(frequencies)
    h2_cache = np.empty((N, Nf), dtype=float)
    fpeak_obs = np.empty(N, dtype=float)

    f = np.asarray(frequencies, dtype=float)

    for k, idxs in enumerate(groups):
        irep = reps[k]

        Mcz_rep = Mcz[irep]
        Mtotz_rep = Mtotz[irep]
        dl_rep = dl[irep]

        h2_rep = rep_h2[k]
        fp_rep = rep_fpeak[k]

        x = f
        y = h2_rep

        for i in idxs:
            amp = (Mcz[i] / Mcz_rep)**(5.0/3.0) * (dl_rep / dl[i])**2

            if use_frequency_warp:
                shift = (Mtotz[i] / Mtotz_rep)  # per-event rescaling factor
                fq = f * shift
                h2_i = amp * _safe_interp_1d(x, y, fq)
                fpeak_obs[i] = fp_rep / shift if np.isfinite(fp_rep) else np.nan
            else:
                h2_i = amp * y
                fpeak_obs[i] = fp_rep

            h2_cache[i, :] = h2_i

    return h2_cache, fpeak_obs


def compute_h2_cache_and_fpeak_parallel(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    frequencies: Optional[np.ndarray] = None,
    nproc: Optional[int] = None,
    chunksize: int = 20,
    *,
    fast_binning: bool = True,
    max_bins: int = 300,
    use_frequency_warp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the waveform-power cache, shape (N_sys, N_freq_eff), and the
    per-event characteristic frequencies, shape (N_sys,).

    Uses the approximate (binned) path when ``fast_binning`` is True,
    otherwise the exact per-event path.
    """
    if frequencies is None:
        wg = build_waveform_generator(wf_cfg)
        frequencies = wg.frequency_array[idx]

    if fast_binning:
        return _compute_fast_binned_h2_cache_and_fpeak(
            dataset=dataset,
            idx=idx,
            wf_cfg=wf_cfg,
            frequencies=np.asarray(frequencies, float),
            nproc=nproc,
            chunksize=int(chunksize),
            max_bins=int(max_bins),
            use_frequency_warp=bool(use_frequency_warp),
        )

    return _compute_exact_h2_cache_and_fpeak_parallel(
        dataset=dataset,
        idx=idx,
        wf_cfg=wf_cfg,
        nproc=nproc,
        chunksize=int(chunksize),
    )


# ============================================================
# Population averages in frequency
# ============================================================
def population_ratio_in_f(
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    weight_per_event: np.ndarray,
) -> np.ndarray:
    """Power-weighted population average of a per-event quantity, as a
    function of frequency."""
    num = np.sum(probabilities[:, None] * h2_cache * weight_per_event[:, None], axis=0)
    den = np.sum(probabilities[:, None] * h2_cache, axis=0)
    return num / den


# ============================================================
# Waveform-level correction
# ============================================================
def compute_Q_ppE(dataset: Dict[str, np.ndarray], a: float, *, alpha_event: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-event amplitude factor used by the waveform-level correction.

    Parameters
    ----------
    a : exponent of the correction term.
    alpha_event : optional per-event multiplicative coefficient of length N_sys.
    """
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    z  = np.asarray(dataset["redshift"], dtype=float)

    chirp_mass   = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    chirp_mass_z = chirp_mass * (1.0 + z)

    # Express the mass in geometric units before forming the factor.
    chirp_mass_z_s = chirp_mass_z * (G.value * M_sun.value / c.value**3)

    Q = (np.pi * chirp_mass_z_s)**(a / 3.0)

    if alpha_event is not None:
        alpha_event = np.asarray(alpha_event, dtype=float)
        if alpha_event.shape != (len(m1),):
            raise ValueError(f"alpha_event must have shape (N_sys,), got {alpha_event.shape}")
        Q = Q * alpha_event

    return Q


def ppe_correction(
    dataset: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    frequencies: np.ndarray,
    a: float,
    alpha_ppE: AlphaLike,
    *,
    f_peak_obs: Optional[np.ndarray] = None,
    f_insp_end_obs: Optional[np.ndarray] = None,
    disable_inspiral_cutoff: bool = False,
    smooth_ppE: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the waveform-level multiplicative correction.

    Returns the per-frequency correction factor and a dict of diagnostics.
    """
    alpha_event = resolve_alpha(alpha_ppE, dataset)
    Q = compute_Q_ppE(dataset, a, alpha_event=alpha_event)
    f = np.asarray(frequencies, dtype=float)

    # Per-event end-of-band frequencies (kept for diagnostics).
    if f_insp_end_obs is None:
        f_insp_end_obs, f_isco_obs, f_isco_src = robust_inspiral_end_frequency_obs(dataset, f_peak_obs=f_peak_obs)
    else:
        f_insp_end_obs = np.asarray(f_insp_end_obs, dtype=float)
        f_isco_src = f_isco_source_hz(dataset)
        z = np.asarray(dataset["redshift"], dtype=float)
        f_isco_obs = f_isco_src / (1.0 + z)

    z = np.asarray(dataset["redshift"], dtype=float)
    f_insp_end_src = f_insp_end_obs * (1.0 + z)

    den = np.sum(probabilities[:, None] * h2_cache, axis=0)

    if bool(disable_inspiral_cutoff):
        # No per-event cutoff: use all frequencies.
        num = np.sum(probabilities[:, None] * h2_cache * Q[:, None], axis=0)
        mask_fraction = np.ones_like(f, dtype=float)
    else:
        mask = (f[None, :] <= f_insp_end_obs[:, None])
        num = np.sum(probabilities[:, None] * h2_cache * (Q[:, None] * mask), axis=0)
        mask_fraction = mask.mean(axis=0)

    R = np.zeros_like(den)
    good = den > 0
    R[good] = num[good] / den[good]

    # Optional smoothing of R(f).
    if smooth_ppE:
        R_smooth = R.copy()
        ok = np.isfinite(R_smooth) & (R_smooth > 0) & np.isfinite(f) & (f > 0)

        if np.sum(ok) > 15:
            x = np.log(f[ok])
            y = np.log(R_smooth[ok])
            x_u = np.linspace(x.min(), x.max(), x.size)
            y_u = np.interp(x_u, x, y)

            w = max(11, int(0.03 * len(y_u)) | 1)
            w = min(w, len(y_u) - (1 - len(y_u) % 2))
            if w < 11:
                w = 11 if len(y_u) >= 11 else (len(y_u) | 1)

            if w >= 5:
                y_u_s = savgol_filter(y_u, window_length=w, polyorder=3, mode="interp")
                y_s = np.interp(x, x_u, y_u_s)
                R_smooth[ok] = np.exp(y_s)
                R = R_smooth

    corr_ppE = 1.0 + 2.0 * f**(a/3.0) * R

    extras = {
        "Q": Q,
        "alpha_event": alpha_event,
        "f_isco_obs": f_isco_obs,
        "f_isco_src": f_isco_src,
        "f_peak_obs": None if f_peak_obs is None else np.asarray(f_peak_obs, dtype=float),
        "f_insp_end_obs": f_insp_end_obs,
        "f_insp_end_src": f_insp_end_src,
        "mask_insp_fraction": mask_fraction,
        "R_insp_over_full": R,
        "disable_inspiral_cutoff": bool(disable_inspiral_cutoff),
    }
    return corr_ppE, extras


# ============================================================
# Propagation/coupling-level correction
# ============================================================
def G_correction(
    dataset: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    h2_cache: np.ndarray,
    G_event_weight: EventWeightFn,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the propagation/coupling-level correction from a per-event
    weight function. Returns the per-frequency factor and diagnostics."""
    G_w = G_event_weight(dataset)
    if not isinstance(G_w, np.ndarray):
        G_w = np.asarray(G_w)

    N_sys = len(dataset["mass_1"])
    if G_w.shape != (N_sys,):
        raise ValueError(f"G_event_weight(dataset) must return shape (N_sys,), got {G_w.shape}")

    corr_G = population_ratio_in_f(probabilities, h2_cache, G_w)
    return corr_G, {"G_weight": G_w}


# ============================================================
# Orchestrator
# ============================================================
def compute_correction(
    dataset: Dict[str, np.ndarray],
    Lambda: Dict[str, float],
    ctx: CorrectionContext,
    *,
    a: float,
    alpha_ppE: AlphaLike,
    G_event_weight: EventWeightFn,
    nproc: Optional[int] = None,
    chunksize: int = 20,
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_f_peak_obs: Optional[np.ndarray] = None,
    precomputed_probabilities: Optional[np.ndarray] = None,
    # approximation knobs:
    fast_binning: bool = True,
    max_bins: int = 250,
    use_frequency_warp: bool = True,
    # optional override of the cutoff behavior without changing wf_cfg
    disable_inspiral_cutoff: Optional[bool] = None,
    smooth_ppE: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Orchestrate the full correction.

    Assembles the population weights and the waveform-power cache (or reuses
    precomputed ones), then combines the available correction terms. Returns
    the two per-frequency correction factors and a dict of diagnostics.
    """
    models = ctx.models

    probabilities = (
        precomputed_probabilities
        if precomputed_probabilities is not None
        else calculate_probabilities(dataset, Lambda, models)
    )

    if precomputed_h2_cache is None or precomputed_f_peak_obs is None:
        h2_cache, f_peak_obs = compute_h2_cache_and_fpeak_parallel(
            dataset=dataset,
            idx=ctx.idx_freq,
            wf_cfg=ctx.wf_cfg,
            frequencies=ctx.frequencies,
            nproc=nproc,
            chunksize=chunksize,
            fast_binning=fast_binning,
            max_bins=max_bins,
            use_frequency_warp=use_frequency_warp,
        )
    else:
        h2_cache = precomputed_h2_cache
        f_peak_obs = precomputed_f_peak_obs

    # Resolve the cutoff toggle: default from wf_cfg unless overridden here.
    _disable_cut = (
        bool(getattr(ctx.wf_cfg, "disable_inspiral_cutoff", False))
        if disable_inspiral_cutoff is None
        else bool(disable_inspiral_cutoff)
    )

    corr_ppE, extras_ppE = ppe_correction(
        dataset=dataset,
        probabilities=probabilities,
        h2_cache=h2_cache,
        frequencies=ctx.frequencies,
        a=a,
        alpha_ppE=alpha_ppE,
        f_peak_obs=f_peak_obs,
        disable_inspiral_cutoff=_disable_cut,
        smooth_ppE = smooth_ppE,
    )

    corr_G, extras_G = G_correction(
        dataset=dataset,
        probabilities=probabilities,
        h2_cache=h2_cache,
        G_event_weight=G_event_weight,
    )

    extras = {
        "models": models,
        "frequencies": ctx.frequencies,
        "idx_freq": ctx.idx_freq,
        "probabilities": probabilities,
        "h2_cache": h2_cache,
        "f_peak_obs": f_peak_obs,
        "fast_binning": fast_binning,
        "max_bins": max_bins,
        "use_frequency_warp": use_frequency_warp,
        "inspiral_only": bool(getattr(ctx.wf_cfg, "inspiral_only", False)),
        "disable_inspiral_cutoff_effective": bool(_disable_cut),
        **extras_ppE,
        **extras_G,
    }

    return corr_ppE, corr_G, extras
