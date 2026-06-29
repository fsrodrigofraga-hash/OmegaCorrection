# SimulatedSignal.py
# ============================================================
# SGWB (cross-correlation) simulator with ppE corrections using
# CorrectionOmegaGW, following the structure of the upstream
# example noncommutative_sweep.py.
#
# - Precompute: context + PopStock omega(fid) + dataset + probabilities
# - build_injected_omega: applies compute_correction (ppE + G)
# - simulate_analytical: closed-form Omega_hat(f) = Omega_true(f) + noise
#                        on the final rebinned grid (no time-domain
#                        simulation, no Welch rebinning).
# - save_npz: writes spectral arrays and basic metadata to a .npz file
#
# Default detectors: CE (Cosmic Explorer 40 km) + ET1/ET2/ET3
# (Einstein Telescope). Det3/det4 are optional — when absent a single
# CE-ET1 baseline is used, preserving backwards compatibility.
# ============================================================

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, Tuple, Union

import numpy as np

import bilby

from pygwb.constants import H0

import OmegaGW_MG as corr

from popstock.PopulationOmegaGW import PopulationOmegaGW
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift


# ============================================================
# Types
# ============================================================
GWeightFn = Callable[[Dict[str, Any]], np.ndarray]
AlphaPPE = Union[float, Callable[[Dict[str, Any]], np.ndarray]]


# ============================================================
# Configs
# ============================================================
@dataclass(frozen=True)
class FrequencySettings:
    fmin: float = 10.0
    fmax: float = 2048.0
    n_freq_eff: int = 800          # N_freq_eff
    n_proposal_samples: int = 10000
    rebin_df_hz: float = 0.5
    gamma_min: float = 0.10
    floor: float = 1e-60


@dataclass(frozen=True)
class InjectionSettings:
    a_true: float = 2.0
    alpha_true: float = 4e-3       # used when alpha_ppE is not provided


@dataclass(frozen=True)
class DetectorSettings:
    # Primary and secondary detectors (required — define the main baseline)
    det1_name: str = "CE"       # Cosmic Explorer 40 km
    det2_name: str = "ET1"      # Einstein Telescope, arm 1
    # Additional detectors (optional — when present they enable the Network)
    det3_name: Optional[str] = None   # e.g., "ET2"
    det4_name: Optional[str] = None   # e.g., "ET3"
    placeholder_if_missing: bool = True
    # WARNING: psd_scale != 1.0 multiplies the detector PSD and alters the
    # noise level artificially. Default 1.0 = real physical detector.
    psd_scale: float = 1.0

    # Einstein Telescope site — the only field the user has to touch.
    # "sardinia" (default) or "meuse". The physical parameters
    # (coordinates, arm length, PSD) are managed internally.
    et_site: str = "sardinia"    # "sardinia" | "meuse"


@dataclass(frozen=True)
class TimeDomainSettings:
    duration: int = 64
    n_segs: int = 5
    fs: int = 4096
    seed_noise: int = 123
    seed_signal: int = 999


@dataclass(frozen=True)
class WelchSettings:
    fft_length: int = 32
    overlap: int = 16


@dataclass(frozen=True)
class PopulationSettings:
    binary_type: str = "BBH"
    waveform_approximant: str = "IMRPhenomD"
    inspiral_only: bool = True
    disable_inspiral_cutoff: bool = True
    minimum_frequency: float = 10.0
    sampling_frequency: float = 4096.0
    duration: float = 4.0

    # gwpopulation models
    mmin_internal: float = 2.0
    mmax_internal: float = 100.0
    z_max: float = 10.0

    # compute_correction controls (as in the upstream example)
    chunksize: int = 20
    fast_binning: bool = True
    max_bins: int = 250
    use_frequency_warp: bool = True


# ============================================================
# PopStock helpers
# ============================================================
def _infer_required_kwargs(callable_obj) -> list[str]:
    try:
        sig = inspect.signature(callable_obj)
    except TypeError:
        sig = inspect.signature(callable_obj.__call__)

    params = list(sig.parameters.values())
    nonself = [p for p in params if p.name != "self"]
    if len(nonself) >= 1:
        nonself = nonself[1:]  # drop dataset

    names = []
    for p in nonself:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        names.append(p.name)
    return names


def _ensure_popstock_args_and_fiducial(pop: PopulationOmegaGW, Lambda: Dict[str, float]) -> Dict[str, float]:
    if not hasattr(pop, "model_args") or pop.model_args is None:
        pop.model_args = {}

    mass_obj = pop.models["mass"]
    red_obj = pop.models["redshift"]

    mass_keys = _infer_required_kwargs(mass_obj)
    red_keys = _infer_required_kwargs(red_obj)

    if len(mass_keys) == 0:
        mass_keys = ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "beta", "delta_m"]
    if len(red_keys) == 0:
        red_keys = ["gamma", "kappa", "z_peak"]

    pop.model_args["mass"] = mass_keys
    pop.model_args["redshift"] = red_keys

    fid = {}
    for k in mass_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    for k in red_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    if "rate" in Lambda:
        fid["rate"] = Lambda["rate"]

    for attr in ("fiducial_parameters", "fiducial_params", "fiducial_parameter", "_fiducial_parameters"):
        if hasattr(pop, attr):
            try:
                setattr(pop, attr, fid)
            except Exception:
                pass

    return fid


def pop_draw_samples(pop: PopulationOmegaGW, Lambda: Dict[str, float], N: int, seed: int) -> Any:
    np.random.seed(int(seed))
    fid = _ensure_popstock_args_and_fiducial(pop, Lambda)

    sig = inspect.signature(pop.draw_and_set_proposal_samples)

    for method in ("direct", "grid"):
        try:
            kwargs = dict(N_proposal_samples=int(N), mass=method, redshift=method)
            if "seed" in sig.parameters:
                kwargs["seed"] = int(seed)
            return pop.draw_and_set_proposal_samples(fid, **kwargs)
        except UnboundLocalError:
            continue
        except TypeError:
            try:
                kwargs = dict(N_proposal_samples=int(N))
                if "seed" in sig.parameters:
                    kwargs["seed"] = int(seed)
                return pop.draw_and_set_proposal_samples(fid, **kwargs)
            except Exception:
                continue

    raise RuntimeError("Failed to draw proposal samples in PopStock with methods ('direct', 'grid').")


def _call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)


def pop_calculate_omega(
    pop: PopulationOmegaGW,
    Lambda: Dict[str, float],
    waveform_approximant: str,
    sampling_frequency: float,
    waveform_minimum_frequency: float,
    multiprocess: bool = False,
) -> Any:
    sig = inspect.signature(pop.calculate_omega_gw)
    kw = dict(
        waveform_approximant=waveform_approximant,
        sampling_frequency=sampling_frequency,
        waveform_minimum_frequency=waveform_minimum_frequency,
        minimum_frequency=waveform_minimum_frequency,
        multiprocess=multiprocess,
    )
    if "Lambda" in sig.parameters:
        return _call_with_supported_kwargs(pop.calculate_omega_gw, Lambda=Lambda, **kw)
    return _call_with_supported_kwargs(pop.calculate_omega_gw, Lambda, **kw)


# ============================================================
# Main class
# ============================================================
class SimulatedSignal:
    """
    Main API:

      models = {"mass": ..., "redshift": ...}
      ctx = build_context(...)
      pop = PopulationOmegaGW(models=models, frequency_array=ctx.frequencies)
      pop_draw_samples(...)
      pop_calculate_omega(...)
      dataset = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)
      probabilities = pop.calculate_probabilities(dataset, Lambda)
      corr_ppE, corr_G, extras = corr.compute_correction(...)

    """

    def __init__(
        self,
        freq: FrequencySettings = FrequencySettings(),
        inj: InjectionSettings = InjectionSettings(),
        det: DetectorSettings = DetectorSettings(),
        td: TimeDomainSettings = TimeDomainSettings(),
        welch: WelchSettings = WelchSettings(),
        popset: PopulationSettings = PopulationSettings(),
        Lambda: Optional[Dict[str, float]] = None,
    ):
        self.freq = freq
        self.inj = inj
        self.det = det
        self.td = td
        self.welch = welch
        self.popset = popset

        if Lambda is None:
            Lambda = {
                "alpha": 2.5, "beta": 1.0, "delta_m": 3.0, "lam": 0.04,
                "mmin": 5.0, "mmax": 100.0, "mpp": 33.0, "sigpp": 5.0,
                "gamma": 2.7, "kappa": 5.0, "z_peak": 1.9,
                "rate": 15.0,
            }
        self.Lambda = dict(Lambda)

        self._models: Optional[Dict[str, Any]] = None
        self._ctx: Optional[Any] = None
        self._pop: Optional[PopulationOmegaGW] = None
        self._dataset: Optional[Dict[str, Any]] = None
        self._omega_fid: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None

    # -------------------------
    # Context construction
    # -------------------------
    def make_models(self) -> Dict[str, Any]:
        return {
            "mass": SinglePeakSmoothedMassDistribution(
                mmin=float(self.popset.mmin_internal),
                mmax=float(self.popset.mmax_internal),
            ),
            "redshift": MadauDickinsonRedshift(z_max=float(self.popset.z_max)),
        }

    def prepare_context_with_models(self, models, wf_cfg, fgrid_cfg):
        wg = corr.build_waveform_generator(wf_cfg)
        full_f = wg.frequency_array
        frequencies, idx = corr.build_frequency_grid(full_f, fgrid_cfg)
        return corr.CorrectionContext(
            models=models,
            wf_cfg=wf_cfg,
            fgrid_cfg=fgrid_cfg,
            frequencies=frequencies,
            idx_freq=idx,
        )

    def build_context(self) -> Any:
        wf_cfg = corr.WaveformConfig(
            binary_type=str(self.popset.binary_type),
            waveform_approximant=str(self.popset.waveform_approximant),
            inspiral_only=bool(self.popset.inspiral_only),
            disable_inspiral_cutoff=bool(self.popset.disable_inspiral_cutoff),
            minimum_frequency=float(self.freq.fmin),
            sampling_frequency=float(self.popset.sampling_frequency),
            duration=float(self.popset.duration),
        )
        fgrid_cfg = corr.FreqGridConfig(
            fmin=float(self.freq.fmin),
            fmax=float(self.freq.fmax),
            N_freq_eff=int(self.freq.n_freq_eff),
        )
        models = self.make_models()
        return self.prepare_context_with_models(models, wf_cfg, fgrid_cfg)

    # -------------------------
    # Precompute pipeline
    # -------------------------
    def precompute_population(self) -> None:
        ctx = self.build_context()
        models = ctx.models

        pop = PopulationOmegaGW(models=models, frequency_array=ctx.frequencies)
        pop_draw_samples(pop, self.Lambda, N=int(self.freq.n_proposal_samples), seed=int(self.td.seed_signal))

        pop_calculate_omega(
            pop,
            Lambda=self.Lambda,
            waveform_approximant=str(self.popset.waveform_approximant),
            sampling_frequency=float(self.popset.sampling_frequency),
            waveform_minimum_frequency=float(self.freq.fmin),
            multiprocess=False,
        )

        dataset = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)
        omega = np.asarray(getattr(pop, "omega_gw"), dtype=float)
        omega = np.where(np.isfinite(omega) & (omega >= 0), omega, 0.0)

        # Probabilities: same pattern as the upstream example
        # (pop.calculate_probabilities).
        pop_tmp = PopulationOmegaGW(models=models)
        _ensure_popstock_args_and_fiducial(pop_tmp, self.Lambda)
        probabilities = pop_tmp.calculate_probabilities(dataset, self.Lambda)
        probabilities = np.asarray(probabilities, dtype=float)

        self._ctx = ctx
        self._models = models
        self._pop = pop
        self._dataset = dataset
        self._omega_fid = omega
        self._probabilities = probabilities

    def _ensure_precomputed(self):
        if self._ctx is None:
            self.precompute_population()

    # -------------------------
    # Correction application
    # -------------------------
    @staticmethod
    def _unity_G(dataset: Dict[str, Any]) -> np.ndarray:
        return np.ones_like(np.asarray(dataset["redshift"], dtype=float), dtype=float)

    def build_injected_omega(
        self,
        a: Optional[float] = None,
        alpha_ppE: Optional[AlphaPPE] = None,
        *,
        G_event_weight: Optional[GWeightFn] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns omega_inj(f) and the extras dict from compute_correction.
        - alpha_ppE may be a float OR a callable(dataset) -> array(N_sys,)
        - G_event_weight must be a callable(dataset) -> array(N_sys,)
        """
        self._ensure_precomputed()

        a_use = float(self.inj.a_true if a is None else a)
        alpha_use = self.inj.alpha_true if alpha_ppE is None else alpha_ppE

        if G_event_weight is None:
            G_event_weight = self._unity_G

        # Enforce correct shapes (N_sys,)
        def _wrap_G(ds):
            w = np.asarray(G_event_weight(ds), dtype=float)
            n = len(np.asarray(ds["redshift"]))
            if w.shape != (n,):
                raise ValueError(f"G_event_weight must return shape (N_sys,), got {w.shape}, expected ({n},)")
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            return w

        if callable(alpha_use):
            def _alpha_callable(ds):
                v = np.asarray(alpha_use(ds), dtype=float)
                n = len(np.asarray(ds["redshift"]))
                if v.shape != (n,):
                    raise ValueError(f"alpha_ppE callable must return shape (N_sys,), got {v.shape}, expected ({n},)")
                v = np.where(np.isfinite(v), v, 0.0)
                return v
            alpha_arg = _alpha_callable
        else:
            alpha_arg = float(alpha_use)

        corr_ppE, corr_G, extras = corr.compute_correction(
            dataset=self._dataset,
            Lambda=self.Lambda,
            ctx=self._ctx,
            a=a_use,
            alpha_ppE=alpha_arg,
            G_event_weight=_wrap_G,
            precomputed_probabilities=self._probabilities,
            nproc=None,
            chunksize=int(self.popset.chunksize),
            fast_binning=bool(self.popset.fast_binning),
            max_bins=int(self.popset.max_bins),
            use_frequency_warp=bool(self.popset.use_frequency_warp),
        )

        corr_ppE = np.asarray(corr_ppE, dtype=float)
        corr_G = np.asarray(corr_G, dtype=float)

        omega_inj = self._omega_fid * corr_ppE * corr_G
        omega_inj = np.where(np.isfinite(omega_inj) & (omega_inj >= 0), omega_inj, 0.0)

        return omega_inj, extras

    # ------------------------------------------------------------------
    # Einstein Telescope name mapping
    # ------------------------------------------------------------------
    # bilby.gw.detector.get_empty_interferometer does NOT know "ET1"/"ET2"/"ET3".
    # ET is modelled in bilby via TriangularInterferometer, which creates the
    # three arms internally. The physical parameters of each site are fixed
    # and managed here — the user only chooses et_site in DetectorSettings.
    #
    # Sources:
    #   Sardinia — ET-D Design Report 2020, site A (Sos Enattos)
    #   Meuse    — ET-D Design Report 2020, site B (Euregio Meuse-Rhine)
    _ET_ALIASES: dict = {"ET1": 0, "ET2": 1, "ET3": 2}

    _ET_SITE_PARAMS: dict = {
        # (latitude_N_deg, longitude_E_deg, length_km, xarm_azimuth_deg)
        "sardinia": (40.5209, 9.4263, 10.0, 70.5674),
        "meuse":    (50.7259, 6.0080, 10.0, 75.0000),
    }

    def _et_params(self) -> tuple:
        """Return (lat, lon, length_km, xarm_az) for the configured ET site."""
        site = str(self.det.et_site).lower()
        if site not in self._ET_SITE_PARAMS:
            raise ValueError(
                f"et_site='{site}' unknown. "
                f"Valid options: {list(self._ET_SITE_PARAMS)}"
            )
        return self._ET_SITE_PARAMS[site]

    @staticmethod
    def _bilby_psd_from_noise_curves(filename: str):
        """
        Load a PSD/ASD from bilby's noise_curves directory.
        Picks psd_file vs asd_file automatically by filename extension.
        Returns a PowerSpectralDensity or None if the file does not exist.
        """
        import os
        bilby_dir = os.path.dirname(bilby.gw.detector.__file__)
        curves_dir = os.path.join(bilby_dir, "noise_curves")
        path = os.path.join(curves_dir, filename)
        if not os.path.exists(path):
            return None
        if "asd" in filename.lower():
            return bilby.gw.detector.PowerSpectralDensity(asd_file=path)
        return bilby.gw.detector.PowerSpectralDensity(psd_file=path)

    def _et_psd(self):
        """
        ET-D PSD: tries to load ET_D_psd.txt from bilby.
        Falls back to aLIGO if the file is not available.
        """
        psd = self._bilby_psd_from_noise_curves("ET_D_psd.txt")
        if psd is None:
            # Fallback — warn but do not interrupt.
            import warnings
            warnings.warn(
                "ET_D_psd.txt not found in bilby. "
                "Using aLIGO as a proxy (less accurate).",
                RuntimeWarning,
            )
            psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        return psd

    def _get_or_make_ifo(self, name: str, allow_placeholder: bool):
        # Special case: Einstein Telescope arms (ET1 / ET2 / ET3)
        if name in self._ET_ALIASES:
            idx = self._ET_ALIASES[name]
            lat, lon, length, xarm_az = self._et_params()
            try:
                tri = bilby.gw.detector.TriangularInterferometer(
                    name="ET",
                    power_spectral_density=self._et_psd(),   # correct ET-D PSD
                    minimum_frequency=float(self.freq.fmin),
                    maximum_frequency=float(self.freq.fmax),
                    length=length,
                    latitude=lat,
                    longitude=lon,
                    elevation=0.0,
                    xarm_azimuth=xarm_az,
                    yarm_azimuth=xarm_az + 60.0,
                )
                ifo = tri[idx]
                ifo.name = name
                return ifo, False
            except Exception:
                if not allow_placeholder:
                    raise
                ifo = bilby.gw.detector.get_empty_interferometer("H1")
                ifo.name = name
                return ifo, True

        # General case: detectors known to bilby (CE, H1, L1, V1, ...).
        # get_empty_interferometer("CE") already loads the CE PSD natively.
        try:
            ifo = bilby.gw.detector.get_empty_interferometer(name)
            return ifo, False
        except Exception:
            if not allow_placeholder:
                raise
            ifo = bilby.gw.detector.get_empty_interferometer("L1")
            ifo.name = name
            return ifo, True

    def _make_ifos(self):
        # Build the IFO list from the det1..det4 fields of DetectorSettings.
        # det1 and det2 are always created; det3 and det4 are optional.
        names = [self.det.det1_name, self.det.det2_name]
        if self.det.det3_name:
            names.append(self.det.det3_name)
        if self.det.det4_name:
            names.append(self.det.det4_name)

        ifos = []
        for name in names:
            ifo, _ = self._get_or_make_ifo(name, self.det.placeholder_if_missing)
            ifos.append(ifo)

        for ifo in ifos:
            ifo.duration = int(self.td.duration)
            ifo.sampling_frequency = int(self.td.fs)

            f = ifo.frequency_array.astype(float)
            psd = ifo.power_spectral_density_array.astype(float).copy()

            bad = (~np.isfinite(psd)) | (psd <= 0)
            if bad.any() and (~bad).any():
                psd[bad] = np.interp(f[bad], f[~bad], psd[~bad])

            psd *= float(self.det.psd_scale)
            psd = np.where(np.isfinite(psd) & (psd > 0), psd, self.freq.floor)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(f, psd)

        return ifos

    # ------------------------------------------------------------------
    # Overlap reduction function (ORF)
    # ------------------------------------------------------------------
    # Used inside simulate_analytical to compute gamma(f) for the
    # primary baseline (ifos[0], ifos[1]).
    # ------------------------------------------------------------------
    def _compute_orf(self, ifo_a, ifo_b, freqs: np.ndarray) -> np.ndarray:
        """Compute the ORF gamma(f) for a pair of bilby interferometers."""
        from pygwb.orfs import calc_orf
        import inspect

        def _vertex(ifo):
            if hasattr(ifo, "geometry") and hasattr(ifo.geometry, "vertex"):
                return np.asarray(ifo.geometry.vertex, dtype=float)
            return np.asarray(ifo.vertex, dtype=float)

        def _arm(ifo, which):
            if hasattr(ifo, "geometry"):
                for nm in (which, f"{which}_arm", f"{which}arm"):
                    if hasattr(ifo.geometry, nm):
                        return np.asarray(getattr(ifo.geometry, nm), dtype=float)
            for nm in (which, f"{which}_arm", f"{which}arm"):
                if hasattr(ifo, nm):
                    return np.asarray(getattr(ifo, nm), dtype=float)
            raise AttributeError(f"Arm {which} not found in {ifo.name}")

        pos1, pos2 = _vertex(ifo_a), _vertex(ifo_b)
        x1, x2 = _arm(ifo_a, "x"), _arm(ifo_b, "x")
        y1, y2 = _arm(ifo_a, "y"), _arm(ifo_b, "y")
        freqs = np.asarray(freqs, dtype=float)

        sig = inspect.signature(calc_orf)
        params = list(sig.parameters.keys())
        try:
            if params and "frequencies" in params[0].lower():
                orf = calc_orf(freqs, pos1, pos2, x1, x2, y1, y2)
            else:
                orf = calc_orf(pos1, pos2, x1, x2, y1, y2, freqs)
        except Exception:
            orf = calc_orf(freqs, pos1, pos2, x1, x2, y1, y2)

        orf = np.asarray(orf, dtype=float)
        return np.where(np.isfinite(orf), orf, 0.0)

    # ------------------------------------------------------------------
    # Analytical signal model + noise on the final rebinned grid
    # ------------------------------------------------------------------
    def simulate_analytical(
        self,
        omega_inj: np.ndarray,
        T_obs: float = 365.25 * 24 * 3600,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Analytical Omega_hat(f) directly on the final (rebinned) grid:
            Omega_hat(f) = Omega_true(f) + N(0, sigma(f))   if add_noise=True
            Omega_hat(f) = Omega_true(f)                    if add_noise=False

        Frequency-grid choice
        ---------------------
        Everything is computed directly on the final uniform grid with
        spacing rebin_df_hz, using df = rebin_df_hz in the Allen–Romano
        formula. No further rebinning is needed and the resulting sigma
        is consistent with the chosen bin width.

        Parameters
        ----------
        omega_inj : array
            Injected energy spectrum Omega_GW(f) on the grid of
            self._ctx.frequencies.
        T_obs : float
            Observation time in seconds (default: 1 year).
        add_noise : bool
            If True (default), adds Gaussian noise to the signal.
            If False, returns the clean signal — useful for validating
            emulator bias.
        """
        self._ensure_precomputed()

        ifos  = self._make_ifos()                       # used only to obtain PSDs and ORF
        f_ifo = ifos[0].frequency_array.astype(float)

        # PSDs of the two primary detectors (on the dense IFO grid)
        P1_ifo = ifos[0].power_spectral_density_array.astype(float)
        P2_ifo = ifos[1].power_spectral_density_array.astype(float)
        P1_ifo = np.where(np.isfinite(P1_ifo) & (P1_ifo > 0), P1_ifo, self.freq.floor)
        P2_ifo = np.where(np.isfinite(P2_ifo) & (P2_ifo > 0), P2_ifo, self.freq.floor)

        # === FINAL GRID DIRECTLY — no rebinning ===
        df_bin = float(self.freq.rebin_df_hz)
        fmin   = float(self.freq.fmin)
        fmax   = float(self.freq.fmax)
        f_bin_target = np.arange(fmin, fmax + 0.5 * df_bin, df_bin)

        # PSDs interpolated onto the final grid
        P1 = np.interp(f_bin_target, f_ifo, P1_ifo,
                       left=self.freq.floor, right=self.freq.floor)
        P2 = np.interp(f_bin_target, f_ifo, P2_ifo,
                       left=self.freq.floor, right=self.freq.floor)
        P1 = np.where(np.isfinite(P1) & (P1 > 0), P1, self.freq.floor)
        P2 = np.where(np.isfinite(P2) & (P2 > 0), P2, self.freq.floor)

        # ORF gamma(f) on the final grid
        gamma  = self._compute_orf(ifos[0], ifos[1], f_bin_target)
        good   = np.abs(gamma) >= float(self.freq.gamma_min)
        g_safe = np.where(good, np.abs(gamma), np.nan)

        # Analytical sigma (Allen–Romano) with df = final bin width
        H0_SI = float(H0.si.value)
        conv  = (10.0 * np.pi**2) / (3.0 * H0_SI**2)
        f3    = np.where((f_bin_target > 0) & np.isfinite(f_bin_target),
                         f_bin_target**3, np.nan)

        sig_Om = conv * f3 * np.sqrt(P1 * P2 / (2.0 * T_obs * df_bin)) / g_safe

        # Omega_true interpolated onto the final grid
        omega_true = np.interp(
            f_bin_target,
            np.asarray(self._ctx.frequencies, dtype=float),
            np.asarray(omega_inj, dtype=float),
            left=0.0, right=0.0,
        )

        # Omega_hat = true signal + Gaussian noise (optional)
        if add_noise:
            rng   = np.random.default_rng(int(self.td.seed_signal))
            noise = rng.normal(0.0, np.where(np.isfinite(sig_Om), sig_Om, 0.0))
            Om_full = omega_true + noise
        else:
            Om_full = omega_true.copy()

        # Valid band
        band = (
            np.isfinite(Om_full) & np.isfinite(sig_Om)
            & (sig_Om > 0) & (sig_Om < np.inf) & good
        )
        f_bin    = f_bin_target[band]
        Om_bin   = Om_full[band]
        sOm_bin  = sig_Om[band]
        P1_bin   = P1[band]
        P2_bin   = P2[band]
        gamma_bin = gamma[band]

        # raw == bin (no separate rebin step; both outputs kept for compatibility)
        f_raw, Om_raw, sOm_raw = f_bin.copy(), Om_bin.copy(), sOm_bin.copy()

        meta = {
            "freq":                  asdict(self.freq),
            "inj":                   asdict(self.inj),
            "det":                   asdict(self.det),
            "td":                    asdict(self.td),
            "welch":                 asdict(self.welch),
            "popset":                asdict(self.popset),
            "Lambda":                dict(self.Lambda),
            "binary_type":           str(self.popset.binary_type),
            "waveform_approximant":  str(self.popset.waveform_approximant),
            "corr_module":           "CorrectionOmegaGW",
            "estimation_mode":       "analytical",
            "T_obs_yr":              float(T_obs / (365.25 * 24 * 3600)),
            "add_noise":             bool(add_noise),
            "df_bin_hz":             float(df_bin),
            "n_detectors":           len(ifos),
            "n_baselines":           1,
            "detector_names":        [ifo.name for ifo in ifos],
        }
        return f_raw, Om_raw, sOm_raw, f_bin, Om_bin, sOm_bin, P1_bin, P2_bin, gamma_bin, meta

    def save_npz(
        self,
        outpath: str,
        *,
        f_raw: np.ndarray,
        Om_raw: np.ndarray,
        sOm_raw: np.ndarray,
        f_bin: np.ndarray,
        Om_bin: np.ndarray,
        sOm_bin: np.ndarray,
        P1_bin: Optional[np.ndarray] = None,
        P2_bin: Optional[np.ndarray] = None,
        gamma_bin: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save spectral arrays as an .npz file.

        P1_bin, P2_bin and gamma_bin are optional — when present they enable
        the optimal-filter likelihood (likelihood_mode="optimal") in
        ParametersEstimator.
        """
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        payload = dict(
            f_raw=np.asarray(f_raw, dtype=float),
            Omega_hat_raw=np.asarray(Om_raw, dtype=float),
            sigma_Omega_raw=np.asarray(sOm_raw, dtype=float),
            f_bin=np.asarray(f_bin, dtype=float),
            Omega_hat_bin=np.asarray(Om_bin, dtype=float),
            sigma_Omega_bin=np.asarray(sOm_bin, dtype=float),
        )
        if P1_bin is not None:
            payload["P1_bin"]    = np.asarray(P1_bin,    dtype=float)
        if P2_bin is not None:
            payload["P2_bin"]    = np.asarray(P2_bin,    dtype=float)
        if gamma_bin is not None:
            payload["gamma_bin"] = np.asarray(gamma_bin, dtype=float)
        if meta is not None:
            payload["meta"] = np.array([meta], dtype=object)
        np.savez(outpath, **payload)
