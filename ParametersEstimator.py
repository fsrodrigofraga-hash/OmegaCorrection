# ParametersEstimator.py
# ============================================================
# Library-style module for parameter estimation from saved NPZ
# data (typically produced by SimulatedSignal).
#
# Public surface used by the pipelines:
#   - ParamGridSpec, LambdaParamSpec, EstimatorConfig (config dataclasses)
#   - load_npz_flexible
#   - omega_full_from_corrections
#   - estimate_C_pop (Monte-Carlo estimate of the population covariance)
#   - DirectModelEvaluator (evaluates Omega_model(f_bin; theta) on the fly,
#                           with a Lambda-keyed LRU cache for the
#                           probability and denominator arrays)
#   - ParametersEstimator (used by the pipelines as a configuration
#                          container — only __init__, load() and
#                          set_lambda_true_values() are exposed)
#
# Conventions:
#   - alpha_ppE may be a scalar (alpha_event_model=None) or a per-event
#     callable (alpha_event_model=fn(dataset, **alpha_params)).
#   - G_event_weight follows the same scalar/callable pattern.
#
# Notes:
#   - All Lambda-derived caches in DirectModelEvaluator round Lambda
#     values to ``lambda_tol_decimals`` decimals to form the cache key.
#     A larger value (e.g. 4) trades off a tiny mis-attribution risk
#     for many more cache hits during rwalk sampling.
# ============================================================

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from joblib import Parallel, delayed

import OmegaGW_MG as corr


# ============================================================
# Types
# ============================================================
# Signature: fn(dataset, **params) -> array(N_sys,)
EventVectorFn = Callable[..., np.ndarray]


# ============================================================
# Grid spec + config
# ============================================================
@dataclass(frozen=True)
class ParamGridSpec:
    """
    Generic parameter grid spec used to define priors and (historically)
    to lay out precomputed model grids.

    transform:
      - "linear": parameter lives in [min, max].
      - "log10" : parameter lives in [10^min, 10^max] and the axis stores
                  log10(theta). Likelihood parameter is theta itself.
    """
    name: str
    min: float
    max: float
    n: int
    transform: str = "linear"  # "linear" or "log10"
    latex_label: Optional[str] = None


@dataclass(frozen=True)
class LambdaParamSpec:
    """
    Prior specification for a free Lambda hyperparameter.

    name        : exact key as it appears in the Lambda dict (e.g. "alpha", "rate")
    min / max   : prior bounds (in the space defined by ``transform``)
    n           : number of grid points (kept for API compatibility)
    transform   : "linear" or "log10"  (same semantics as ParamGridSpec)
    latex_label : LaTeX string for corner-plot axis labels
    """
    name: str
    min: float
    max: float
    n: int
    transform: str = "linear"
    latex_label: Optional[str] = None


@dataclass(frozen=True)
class EstimatorConfig:
    # Paths
    infile_npz: str = os.path.join("saved_ppE_data", "ppE_simulated_data.npz")
    outdir: str = "outputs_infer_from_saved"
    grid_cache_dir: str = ""

    # Core ppE "a" range
    a_min: float = 0.0
    a_max: float = 10.0
    na: int = 41

    # =========================
    # Scalar alpha_ppE inference mode
    # =========================
    # alpha_scale:
    #   - "log10": LogUniform prior, axis stores log10(alpha)
    #   - "linear": Uniform prior on alpha
    alpha_scale: str = "log10"

    # (A) When alpha_scale == "log10"
    logalpha_min: float = -6.0
    logalpha_max: float = -2.0
    nlogalpha: int = 41

    # (B) When alpha_scale == "linear"
    alpha_min: float = 1e-6
    alpha_max: float = 1e-2
    nalpha: int = 41

    # Cache quantisation for interpolator memoisation (kept for API stability).
    cache_subdiv: int = 10

    # -------------------------
    # Parallel / CPU management
    # -------------------------
    # grid_jobs, sampler_pool and cpop_jobs are auto-detected from
    # cpu_count() when left as None, following a RAM-aware heuristic:
    #   8  GB RAM -> grid_jobs=2, sampler_pool=2, cpop_jobs=1
    #   16 GB RAM -> grid_jobs=4, sampler_pool=4, cpop_jobs=2  (default)
    #   32 GB RAM -> grid_jobs=8, sampler_pool=8, cpop_jobs=4
    # Override any of them explicitly to bypass auto-detection.
    n_jobs: int = -1
    grid_jobs: Optional[int] = None
    sampler_pool: Optional[int] = None
    cpop_jobs: Optional[int] = None

    # Dynesty
    npoints: int = 500
    walks: int = 30
    dlogz: float = 0.1

    # -------------------------
    # C_pop (population covariance)
    # -------------------------
    # k_pop: number of Monte Carlo realisations for C_pop estimation.
    #   Set to 0 to skip C_pop entirely.
    k_pop: int = 50
    pop_seed_base: int = 90000
    cpop_jitter: float = 1e-10

    # -------------------------
    # Likelihood mode
    # -------------------------
    # "diagonal"  - Gaussian, diagonal in sigma_inst (legacy default).
    # "full_cov"  - C_eff = k^2 * C_inst + C_pop (recommended).
    likelihood_mode: str = "full_cov"

    # Numerical floor
    floor: float = 1e-60


# ============================================================
# I/O
# ============================================================
def load_npz_flexible(path: str) -> Dict[str, Any]:
    """
    Load saved arrays. Supports two formats:

    Format 1 (legacy estimator):
      - f_bin, Omega_hat_bin, sigma_Omega_bin, FMIN, FMAX, N_FREQ, N_SYS, REBIN_DF_HZ

    Format 2 (SimulatedSignal.save_npz):
      - f_bin, Omega_hat_bin, sigma_Omega_bin, meta (object array with dict)
        plus optional raw arrays (ignored here)

    Returns a dict containing at least f_bin, Omega_hat_bin, sigma_Omega_bin
    and any available metadata.
    """
    d = np.load(path, allow_pickle=True)

    if "f_bin" not in d.files:
        raise KeyError("NPZ must contain 'f_bin'.")
    if "Omega_hat_bin" not in d.files:
        raise KeyError("NPZ must contain 'Omega_hat_bin'.")
    if "sigma_Omega_bin" not in d.files:
        raise KeyError("NPZ must contain 'sigma_Omega_bin'.")

    out: Dict[str, Any] = {
        "f_bin": d["f_bin"],
        "Omega_hat_bin": d["Omega_hat_bin"],
        "sigma_Omega_bin": d["sigma_Omega_bin"],
    }

    # Noise info for the optimal-filter likelihood (optional)
    for k in ("P1_bin", "P2_bin", "gamma_bin"):
        if k in d.files:
            out[k] = d[k]

    # Legacy metadata
    for k in ("FMIN", "FMAX", "N_FREQ", "N_SYS", "REBIN_DF_HZ"):
        if k in d.files:
            out[k] = d[k]

    # SimulatedSignal meta (object array)
    if "meta" in d.files:
        meta_arr = d["meta"]
        try:
            if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object and meta_arr.size >= 1:
                meta = meta_arr.flat[0]
                if isinstance(meta, dict):
                    out["meta"] = meta
        except Exception:
            pass

    return out


# ============================================================
# Omega(f) from ppE + G corrections
# ============================================================
def omega_full_from_corrections(
    freqs_pop: np.ndarray,
    omega_fid: np.ndarray,
    dataset: Dict[str, Any],
    Lambda: Dict[str, float],
    ctx: Any,
    probabilities: np.ndarray,
    *,
    a: float,
    alpha_ppE: Optional[float] = None,
    alpha_event_model: Optional[EventVectorFn] = None,
    alpha_params: Optional[Dict[str, Any]] = None,
    G_event_weight: Optional[EventVectorFn] = None,
    G_params: Optional[Dict[str, Any]] = None,
    # Pre-computed waveform cache (bypasses LAL calls when provided)
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_f_peak_obs: Optional[np.ndarray] = None,
    # compute_correction knobs
    chunksize: int = 20,
    fast_binning: bool = True,
    max_bins: int = 400,
    use_frequency_warp: bool = True,
    nproc: Optional[int] = 1,
    disable_inspiral_cutoff: Optional[bool] = None,
    smooth_ppE: bool = False,
) -> np.ndarray:
    """
    Apply the ppE + G corrections to a fiducial Omega_GW(f) on the population
    frequency grid and return the corrected spectrum.

    alpha_ppE may be a scalar (alpha_event_model=None) or a callable
    (alpha_event_model=fn(dataset, **alpha_params)). When using the callable
    path, alpha_ppE must be left as None.
    """
    if alpha_params is None:
        alpha_params = {}
    if G_params is None:
        G_params = {}

    if G_event_weight is None:
        def G_event_weight(ds, **params):
            return np.ones(len(ds["mass_1"]), dtype=np.float64)

    def _G_wrap(ds):
        w = G_event_weight(ds, **G_params)
        w = np.asarray(w, dtype=np.float64)
        n_sys = len(ds["mass_1"])
        if w.shape != (n_sys,):
            raise ValueError(f"G_event_weight must return shape (N_sys,), got {w.shape} with N_sys={n_sys}")
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        return w

    if alpha_event_model is not None:
        def _alpha_wrap(ds):
            v = alpha_event_model(ds, **alpha_params)
            v = np.asarray(v, dtype=np.float64)
            n_sys = len(ds["mass_1"])
            if v.shape != (n_sys,):
                raise ValueError(f"alpha_event_model must return shape (N_sys,), got {v.shape} with N_sys={n_sys}")
            v = np.where(np.isfinite(v), v, 0.0)
            return v
        alpha_arg: Union[float, Callable[[Dict[str, Any]], np.ndarray]] = _alpha_wrap
    else:
        if alpha_ppE is None:
            raise ValueError("Provide either alpha_ppE (scalar) or alpha_event_model (callable).")
        alpha_arg = float(alpha_ppE)

    corr_ppE, corr_G, _extras = corr.compute_correction(
        dataset=dataset,
        Lambda=Lambda,
        ctx=ctx,
        a=float(a),
        alpha_ppE=alpha_arg,
        G_event_weight=_G_wrap,
        nproc=nproc,
        chunksize=int(chunksize),
        precomputed_probabilities=probabilities,
        precomputed_h2_cache=precomputed_h2_cache,
        precomputed_f_peak_obs=precomputed_f_peak_obs,
        fast_binning=bool(fast_binning),
        max_bins=int(max_bins),
        use_frequency_warp=bool(use_frequency_warp),
        disable_inspiral_cutoff=disable_inspiral_cutoff,
        smooth_ppE=smooth_ppE,
    )

    corr_ppE = np.asarray(corr_ppE, dtype=np.float64)
    corr_G   = np.asarray(corr_G,   dtype=np.float64)

    omega_full = omega_fid * corr_ppE * corr_G
    omega_full = np.where(np.isfinite(omega_full) & (omega_full >= 0), omega_full, 0.0)
    return omega_full


# ============================================================
# CPU job-count helpers
# ============================================================
def _resolve_cpu_jobs(n_cores: int) -> Tuple[int, int, int]:
    """
    Return (grid_jobs, sampler_pool, cpop_jobs) based on the core count,
    following a RAM-aware heuristic.
    """
    grid_jobs    = min(max(1, n_cores), 8)
    sampler_pool = min(max(1, n_cores), 8)
    cpop_jobs    = max(1, min(n_cores // 4, 4))
    return grid_jobs, sampler_pool, cpop_jobs


def _effective_jobs(cfg_value: Optional[int], auto_value: int) -> int:
    """Return the user-provided override if set, otherwise the auto-detected value."""
    if cfg_value is not None and int(cfg_value) > 0:
        return int(cfg_value)
    return int(auto_value)


# ============================================================
# Ledoit-Wolf shrinkage
# ============================================================
def _ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf OAS shrinkage (via sklearn if available, analytic fallback).
    Optimal for K << n_bins — no manual tuning required.
    Returns (C_shrunk, shrinkage_coeff).
    """
    try:
        from sklearn.covariance import OAS
        oas = OAS()
        oas.fit(X)
        return np.asarray(oas.covariance_, dtype=np.float64), float(oas.shrinkage_)
    except ImportError:
        pass

    # Analytic fallback (Oracle Approximating Shrinkage)
    K, p = X.shape
    S    = np.cov(X, rowvar=False, ddof=1)
    mu   = np.trace(S) / p
    S2   = S @ S
    rho_n = ((K - 2) / K) * np.trace(S2) + np.trace(S) ** 2
    rho_d = (K + 2) * (np.trace(S2) - np.trace(S) ** 2 / p)
    rho   = float(np.clip(rho_n / max(rho_d, 1e-300), 0.0, 1.0))
    C     = (1.0 - rho) * S + rho * mu * np.eye(p)
    return np.asarray(C, dtype=np.float64), rho


# ============================================================
# C_pop Monte Carlo estimation
# ============================================================
def _cpop_worker(
    k_idx: int,
    seed_base: int,
    freq_settings: Any,
    det_settings: Any,
    td_settings: Any,
    welch_settings: Any,
    pop_settings: Any,
    a_ref: float,
    alpha_ref_callable: Optional[Callable],
    alpha_ref_scalar: Optional[float],
    f_bin: np.ndarray,
    # Bootstrap reweighting inputs — avoid rebuilding PopStock per worker
    precomputed_dataset: Optional[Dict[str, Any]] = None,
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_probabilities: Optional[np.ndarray] = None,
    precomputed_omega_fid: Optional[np.ndarray] = None,
    precomputed_freqs: Optional[np.ndarray] = None,
    precomputed_ctx: Optional[Any] = None,
    base_rate: float = 1.0,
    Lambda_base: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """
    Compute Omega_MODEL(f; theta_true) for one population realisation.

    Fast path (bootstrap reweighting):
      When the precomputed_* inputs are provided, this worker bootstraps the
      probabilities to emulate population variance without rebuilding PopStock.
      Cost: O(N_sys * N_freq) per worker — much faster than a full rebuild.

    Legacy path (fallback):
      Rebuilds PopStock from scratch when the precomputed_* inputs are absent.
    """
    seed_k = int(seed_base + k_idx * 1_000_003) % (2 ** 31 - 1)
    rng    = np.random.default_rng(seed_k)

    # ------------------------------------------------------------------
    # Fast path: bootstrap reweighting
    # ------------------------------------------------------------------
    if (precomputed_dataset is not None
            and precomputed_h2_cache is not None
            and precomputed_probabilities is not None
            and precomputed_omega_fid is not None
            and precomputed_freqs is not None
            and precomputed_ctx is not None):
        try:
            N_sys = len(precomputed_probabilities)
            idx_boot   = rng.integers(0, N_sys, size=N_sys)
            probs_boot = precomputed_probabilities[idx_boot]
            h2_boot    = precomputed_h2_cache[idx_boot]

            p_sum = float(np.sum(probs_boot))
            if p_sum <= 0:
                return None
            probs_boot = probs_boot / p_sum * N_sys

            den_boot = np.sum(probs_boot[:, None] * h2_boot, axis=0)
            den_base = np.sum(precomputed_probabilities[:, None] * precomputed_h2_cache, axis=0)
            den_ratio = np.where(
                (den_base > 0) & (den_boot > 0),
                den_boot / den_base, 1.0,
            )

            alpha_arg = alpha_ref_callable if alpha_ref_callable is not None \
                        else float(alpha_ref_scalar or 1e-6)

            omega_fid_boot = precomputed_omega_fid * den_ratio

            # Use the base Lambda — population variance is already captured
            # via the bootstrap of the probability weights.
            Lambda_boot = dict(Lambda_base) if Lambda_base is not None else {}

            omega_k = omega_full_from_corrections(
                precomputed_freqs, omega_fid_boot,
                precomputed_dataset, Lambda_boot,
                precomputed_ctx, probs_boot,
                a=float(a_ref),
                alpha_ppE=alpha_arg,
                precomputed_h2_cache=h2_boot,
                nproc=1,
            )

            omega_k = np.asarray(omega_k, dtype=float)
            omega_k = np.where(np.isfinite(omega_k) & (omega_k >= 0), omega_k, 0.0)
            return np.interp(np.asarray(f_bin, dtype=float),
                             precomputed_freqs, omega_k, left=0.0, right=0.0)
        except Exception as exc:
            import warnings
            warnings.warn(f"[C_pop worker {k_idx}] bootstrap failed ({exc}), "
                          f"falling back to full rebuild.", RuntimeWarning)

    # ------------------------------------------------------------------
    # Legacy path: full rebuild
    # ------------------------------------------------------------------
    from SimulatedSignal import SimulatedSignal, InjectionSettings, TimeDomainSettings

    try:
        inj_k = InjectionSettings(a_true=float(a_ref), alpha_true=float(alpha_ref_scalar or 1e-6))
        td_k  = TimeDomainSettings(
            duration=td_settings.duration,
            n_segs=td_settings.n_segs,
            fs=td_settings.fs,
            seed_noise=int(seed_k + 1),
            seed_signal=int(seed_k),
        )
        sim_k = SimulatedSignal(
            freq=freq_settings,
            inj=inj_k,
            det=det_settings,
            td=td_k,
            welch=welch_settings,
            popset=pop_settings,
        )
        np.random.seed(seed_k)
        sim_k.precompute_population()

        alpha_arg = alpha_ref_callable if alpha_ref_callable is not None \
                    else float(alpha_ref_scalar or 1e-6)

        omega_k, _ = sim_k.build_injected_omega(a=float(a_ref), alpha_ppE=alpha_arg)

        f_model = np.asarray(sim_k._ctx.frequencies, dtype=float)
        omega_k = np.asarray(omega_k, dtype=float)
        omega_k = np.where(np.isfinite(omega_k) & (omega_k >= 0), omega_k, 0.0)
        return np.interp(np.asarray(f_bin, dtype=float), f_model, omega_k,
                         left=0.0, right=0.0)
    except Exception as exc:
        import warnings
        warnings.warn(f"[C_pop worker {k_idx}] failed: {exc}", RuntimeWarning)
        return None


def estimate_C_pop(
    f_bin: np.ndarray,
    a_ref: float,
    *,
    # alpha at injection truth — either scalar or callable(dataset)->array
    alpha_ref_callable: Optional[Callable] = None,
    alpha_ref_scalar: Optional[float] = None,
    # SimulatedSignal settings objects
    freq_settings: Any,
    det_settings: Any,
    td_settings: Any,
    welch_settings: Any,
    pop_settings: Any,
    # MC parameters
    k_pop: int = 50,
    seed_base: int = 90000,
    jitter: float = 1e-10,
    n_jobs: int = 2,
    # Lambda used to pre-compute the base PopStock for bootstrap reweighting
    Lambda: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the population covariance matrix C_pop via K Monte Carlo
    realisations of Omega_MODEL at the injection truth parameters.

    Fast path (bootstrap reweighting):
      Precomputes the base PopStock once, then each worker bootstraps the
      probabilities to emulate population variance.
      Cost: 1 PopStock build + K_POP * O(N_sys * N_freq) — ~K_POP times
      faster than the legacy full-rebuild path.

    Returns (C_pop, omega_samples) where:
      - C_pop has shape (n_bins, n_bins), Ledoit-Wolf shrunk
      - omega_samples has shape (K_ok, n_bins)
    """
    f_bin  = np.asarray(f_bin, dtype=float)
    n_bins = int(f_bin.size)

    # ------------------------------------------------------------------
    # Pre-compute the base PopStock once for the bootstrap fast path.
    # ------------------------------------------------------------------
    precomputed_dataset       = None
    precomputed_h2_cache      = None
    precomputed_probabilities = None
    precomputed_omega_fid     = None
    precomputed_freqs         = None
    precomputed_ctx           = None
    base_rate                 = 1.0

    try:
        from SimulatedSignal import SimulatedSignal, InjectionSettings
        alpha_arg = alpha_ref_callable if alpha_ref_callable is not None \
                    else float(alpha_ref_scalar or 1e-6)

        sim_base = SimulatedSignal(
            freq=freq_settings,
            inj=InjectionSettings(a_true=float(a_ref),
                                  alpha_true=float(alpha_ref_scalar or 1e-6)),
            det=det_settings,
            td=td_settings,
            welch=welch_settings,
            popset=pop_settings,
            Lambda=Lambda,
        )
        sim_base.precompute_population()

        h2_cache, _ = corr.compute_h2_cache_and_fpeak_parallel(
            dataset=sim_base._dataset,
            idx=sim_base._ctx.idx_freq,
            wf_cfg=sim_base._ctx.wf_cfg,
            frequencies=sim_base._ctx.frequencies,
            nproc=1, chunksize=20,
            fast_binning=True, max_bins=100, use_frequency_warp=True,
        )

        omega_base, _ = sim_base.build_injected_omega(
            a=float(a_ref), alpha_ppE=alpha_arg,
        )

        precomputed_dataset       = sim_base._dataset
        precomputed_h2_cache      = h2_cache
        precomputed_probabilities = np.asarray(sim_base._probabilities, dtype=np.float64)
        precomputed_omega_fid     = np.asarray(sim_base._omega_fid, dtype=np.float64)
        precomputed_freqs         = np.asarray(sim_base._ctx.frequencies, dtype=np.float64)
        precomputed_ctx           = sim_base._ctx
        base_rate                 = float((Lambda or {}).get("rate", 1.0))

        print(f"[estimate_C_pop] base PopStock precomputed — "
              f"bootstrap reweighting over {k_pop} workers.", flush=True)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"[estimate_C_pop] precomputation failed ({exc}). "
            f"Falling back to legacy mode (K_POP full rebuilds).",
            RuntimeWarning,
        )

    results_raw = Parallel(n_jobs=int(n_jobs), backend="loky", verbose=0)(
        delayed(_cpop_worker)(
            k_idx=k,
            seed_base=int(seed_base),
            freq_settings=freq_settings,
            det_settings=det_settings,
            td_settings=td_settings,
            welch_settings=welch_settings,
            pop_settings=pop_settings,
            a_ref=float(a_ref),
            alpha_ref_callable=alpha_ref_callable,
            alpha_ref_scalar=alpha_ref_scalar,
            f_bin=f_bin,
            precomputed_dataset=precomputed_dataset,
            precomputed_h2_cache=precomputed_h2_cache,
            precomputed_probabilities=precomputed_probabilities,
            precomputed_omega_fid=precomputed_omega_fid,
            precomputed_freqs=precomputed_freqs,
            precomputed_ctx=precomputed_ctx,
            base_rate=base_rate,
            Lambda_base=Lambda,
        )
        for k in range(int(k_pop))
    )

    omega_list = [r for r in results_raw if r is not None]
    if not omega_list:
        import warnings
        warnings.warn("[estimate_C_pop] all workers failed — returning jitter*I.", RuntimeWarning)
        return np.eye(n_bins, dtype=np.float64) * jitter, np.empty((0, n_bins))

    omega_arr = np.asarray(omega_list, dtype=float)
    omega_ok  = omega_arr[np.all(np.isfinite(omega_arr), axis=1)]
    k_ok      = omega_ok.shape[0]

    if k_ok < 3:
        import warnings
        warnings.warn(f"[estimate_C_pop] only {k_ok}/{k_pop} valid workers — returning jitter*I.",
                      RuntimeWarning)
        return np.eye(n_bins, dtype=np.float64) * jitter, omega_arr

    C_pop, alpha_lw = _ledoit_wolf_shrinkage(omega_ok)

    diag_mean = max(float(np.mean(np.abs(np.diag(C_pop)))), 1e-300)
    C_pop    += np.eye(C_pop.shape[0]) * (jitter * diag_mean)
    C_pop     = np.asarray(C_pop, dtype=np.float64)

    return C_pop, omega_ok


# ============================================================
# Direct on-the-fly Omega_model evaluator (used by the pipelines)
# ============================================================
class DirectModelEvaluator:
    """
    Compute omega_model(f_bin; theta) directly via omega_full_from_corrections.

    Two-level caching strategy:
      - h2_cache and f_peak_obs are computed ONCE at init from the base
        population and held fixed for the entire run. They depend only on
        the dataset (masses / distances), not on Lambda.
      - probabilities are the only quantity that changes with Lambda. On a
        cache miss only calculate_probabilities() is called (~1 ms, pure
        NumPy) instead of a full population redraw (~2-5 s with waveform
        calls).
      - omega_fid stays fixed at the base-population value. This is correct
        because corr_ppE and corr_G are weighted ratios over h2_cache, so
        the correction factors depend on the *relative* change in
        probabilities, not on the absolute value of omega_fid.

    Cache key: tuple of rounded Lambda free-parameter values
    (rounding precision controlled by ``lambda_tol_decimals``).
    """

    def __init__(
        self,
        f_bin: np.ndarray,
        freqs_pop: np.ndarray,
        omega_fid: np.ndarray,
        dataset: Dict[str, Any],
        base_Lambda: Dict[str, float],
        ctx: Any,
        probabilities: np.ndarray,
        *,
        lambda_param_names: List[str],
        alpha_param_names: List[str],
        G_param_names: List[str],
        alpha_event_model: Optional[EventVectorFn],
        G_event_weight: Optional[EventVectorFn],
        compute_knobs: Dict[str, Any],
        pop_cfg: Optional[Dict[str, Any]] = None,
        lambda_tol_decimals: int = 4,
        lru_size: int = 2,
    ):
        self.f_bin               = np.asarray(f_bin,     dtype=np.float64)
        self.freqs_pop           = np.asarray(freqs_pop, dtype=np.float64)
        self.omega_fid           = np.asarray(omega_fid, dtype=np.float64)
        self.dataset             = dataset
        self.base_Lambda         = dict(base_Lambda)
        self.ctx                 = ctx
        self._base_probabilities = np.asarray(probabilities, dtype=np.float64)

        self.lambda_param_names = list(lambda_param_names)
        self.alpha_param_names  = list(alpha_param_names)
        self.G_param_names      = list(G_param_names)
        self.alpha_event_model  = alpha_event_model
        self.G_event_weight     = G_event_weight
        self.compute_knobs      = dict(compute_knobs)
        self.pop_cfg            = dict(pop_cfg) if pop_cfg is not None else {}
        self.lambda_tol_dec     = int(lambda_tol_decimals)
        self.lru_size           = int(lru_size)

        # Pre-compute h2_cache and f_peak_obs once at init.
        # These are independent of Lambda (depend only on masses / distances).
        knobs = self.compute_knobs
        print("[DirectModelEvaluator] Pre-computing h2_cache (once)...", flush=True)
        self._h2_cache, self._f_peak_obs = corr.compute_h2_cache_and_fpeak_parallel(
            dataset=self.dataset,
            idx=self.ctx.idx_freq,
            wf_cfg=self.ctx.wf_cfg,
            frequencies=self.ctx.frequencies,
            nproc=knobs.get("nproc", 1),
            chunksize=int(knobs.get("chunksize", 20)),
            fast_binning=bool(knobs.get("fast_binning", True)),
            max_bins=int(knobs.get("max_bins", 100)),
            use_frequency_warp=bool(knobs.get("use_frequency_warp", True)),
        )
        print("[DirectModelEvaluator] h2_cache ready.", flush=True)

        # den_base(f) = sum_i p(theta_i | Lambda_base) * h_i^2(f)  —
        # fixed denominator used in importance reweighting.
        self._den_base = np.sum(
            np.asarray(probabilities, dtype=np.float64)[:, None] * self._h2_cache,
            axis=0,
        )
        # Base merger-rate value used for rate reweighting.
        self._rate_base = float(base_Lambda.get("rate", 1.0))

        # LRU cache: key -> probabilities array
        self._prob_cache: OrderedDict = OrderedDict()
        # LRU cache: key -> den_novo array
        self._den_cache:  OrderedDict = OrderedDict()

    def _lambda_key(self, theta: Dict[str, float]) -> Tuple[float, ...]:
        """Rounded-value cache key for the free Lambda parameters."""
        return tuple(
            round(float(theta[k]), self.lambda_tol_dec)
            for k in self.lambda_param_names
        )

    def _get_probabilities_for_lambda(self, theta: Dict[str, float]) -> np.ndarray:
        """
        Return the per-event probabilities for the Lambda values in theta.
        On a cache miss only calculate_probabilities() runs (~1 ms, pure
        NumPy); h2_cache stays fixed.
        """
        if not self.lambda_param_names:
            return self._base_probabilities

        key = self._lambda_key(theta)

        if key in self._prob_cache:
            self._prob_cache.move_to_end(key)
            return self._prob_cache[key]

        Lambda_here = dict(self.base_Lambda)
        for k in self.lambda_param_names:
            Lambda_here[k] = float(theta[k])

        probs = np.asarray(
            corr.calculate_probabilities(self.dataset, Lambda_here, self.ctx.models),
            dtype=np.float64,
        )

        if len(self._prob_cache) >= self.lru_size:
            self._prob_cache.popitem(last=False)
        self._prob_cache[key] = probs
        return probs

    def model(self, theta: Dict[str, float]) -> np.ndarray:
        """
        Compute omega_model interpolated onto f_bin for the parameter point theta.
        theta must contain: 'a', all alpha_param_names or 'alpha_ppE', and all
        lambda_param_names.
        """
        probs = self._get_probabilities_for_lambda(theta)

        a            = float(theta.get("a", self.base_Lambda.get("a", 0.0)))
        alpha_params = {k: float(theta[k]) for k in self.alpha_param_names}
        G_params     = {k: float(theta[k]) for k in self.G_param_names}

        if self.alpha_event_model is None:
            alpha_ppE         = float(theta["alpha_ppE"])
            alpha_event_model = None
        else:
            alpha_ppE         = None
            alpha_event_model = self.alpha_event_model

        # Build the per-point Lambda (required by omega_full_from_corrections).
        Lambda_here = dict(self.base_Lambda)
        for k in self.lambda_param_names:
            Lambda_here[k] = float(theta[k])

        # Importance reweighting of omega_fid for the proposed Lambda.
        rate_novo  = float(Lambda_here.get("rate", self._rate_base))
        rate_ratio = rate_novo / self._rate_base if self._rate_base > 0 else 1.0

        key = self._lambda_key(theta)
        if key in self._den_cache:
            self._den_cache.move_to_end(key)
            den_novo = self._den_cache[key]
        else:
            den_novo = np.sum(probs[:, None] * self._h2_cache, axis=0)
            if len(self._den_cache) >= self.lru_size:
                self._den_cache.popitem(last=False)
            self._den_cache[key] = den_novo

        den_ratio = np.where(
            (self._den_base > 0) & (den_novo > 0),
            den_novo / self._den_base,
            1.0,
        )
        omega_fid_eff = self.omega_fid * rate_ratio * den_ratio

        omega_full = omega_full_from_corrections(
            self.freqs_pop, omega_fid_eff, self.dataset, Lambda_here,
            self.ctx, probs,
            a=a,
            alpha_ppE=alpha_ppE,
            alpha_event_model=alpha_event_model,
            alpha_params=alpha_params,
            G_event_weight=self.G_event_weight,
            G_params=G_params,
            # Pass pre-computed waveform cache — no LAL calls needed.
            precomputed_h2_cache=self._h2_cache,
            precomputed_f_peak_obs=self._f_peak_obs,
            **self.compute_knobs,
        )
        return np.interp(self.f_bin, self.freqs_pop, omega_full,
                         left=0.0, right=0.0).astype(np.float64)


# ============================================================
# Configuration container used by the pipelines
# ============================================================
class ParametersEstimator:
    """
    Configuration container used by the production pipelines.

    The pipelines instantiate this class to centralise prior / event-model
    settings, call ``set_lambda_true_values()`` to record injection truths
    for downstream plots, and call ``load()`` to read back the NPZ produced
    by SimulatedSignal. The pipelines then read the attributes
    ``lambda_param_specs``, ``alpha_param_specs``, ``G_param_specs``,
    ``alpha_event_model`` and ``compute_knobs`` to build their own
    DirectModelEvaluator and run the inference.

    Higher-level workflows (grid building, automatic likelihood selection,
    corner-plot helpers) used to live here too but were removed once all
    production pipelines moved to the DirectModelEvaluator + ad-hoc
    likelihood pattern.
    """

    def __init__(
        self,
        config: EstimatorConfig = EstimatorConfig(),
        *,
        # alpha per-event model (optional). If None, infer scalar alpha_ppE.
        alpha_event_model: Optional[EventVectorFn] = None,
        alpha_id: str = "alpha_default",
        alpha_param_specs: Optional[List[ParamGridSpec]] = None,

        # G event weights (optional)
        G_event_weight: Optional[EventVectorFn] = None,
        G_id: str = "G_default_1",
        G_param_specs: Optional[List[ParamGridSpec]] = None,

        # Free Lambda hyperparameters (optional).
        # Each LambdaParamSpec defines the prior/grid for one Lambda key.
        # Keys not listed here are held fixed at the base Lambda values.
        lambda_param_specs: Optional[List["LambdaParamSpec"]] = None,

        # compute_correction knobs
        chunksize: int = 20,
        fast_binning: bool = True,
        max_bins: int = 400,
        use_frequency_warp: bool = True,
        nproc: Optional[int] = 1,
        disable_inspiral_cutoff: Optional[bool] = None,
    ):
        self.cfg = config
        os.makedirs(self.cfg.outdir, exist_ok=True)

        self.alpha_event_model = alpha_event_model
        self.alpha_id = str(alpha_id)
        self.alpha_param_specs = list(alpha_param_specs) if alpha_param_specs is not None else []

        self.G_event_weight = G_event_weight
        self.G_id = str(G_id)
        self.G_param_specs = list(G_param_specs) if G_param_specs is not None else []

        # Free Lambda hyperparameters — additional prior dimensions.
        self.lambda_param_specs: List[LambdaParamSpec] = (
            list(lambda_param_specs) if lambda_param_specs is not None else []
        )
        # Truth values for corner-plot markers; set via set_lambda_true_values().
        self.lambda_true_values: Dict[str, float] = {}

        self.compute_knobs = dict(
            chunksize=int(chunksize),
            fast_binning=bool(fast_binning),
            max_bins=int(max_bins),
            use_frequency_warp=bool(use_frequency_warp),
            nproc=nproc,
            disable_inspiral_cutoff=disable_inspiral_cutoff,
        )

        # Resolve CPU counts once at construction.
        n_cores = cpu_count() or 1
        _auto_grid, _auto_pool, _auto_cpop = _resolve_cpu_jobs(n_cores)
        self._grid_jobs    = _effective_jobs(config.grid_jobs,    _auto_grid)
        self._sampler_pool = _effective_jobs(config.sampler_pool, _auto_pool)
        self._cpop_jobs    = _effective_jobs(config.cpop_jobs,    _auto_cpop)

        # Loaded data
        self.saved: Optional[Dict[str, Any]] = None
        self.f_bin: Optional[np.ndarray] = None
        self.Om_bin: Optional[np.ndarray] = None
        self.sOm_bin: Optional[np.ndarray] = None
        # Noise info for the optimal-filter likelihood (kept for API stability).
        self.P1_bin:    Optional[np.ndarray] = None
        self.P2_bin:    Optional[np.ndarray] = None
        self.gamma_bin: Optional[np.ndarray] = None

    # -------------------------
    # Load
    # -------------------------
    def load(self, infile_npz: Optional[str] = None) -> Dict[str, Any]:
        path = self.cfg.infile_npz if infile_npz is None else infile_npz
        saved = load_npz_flexible(path)

        self.saved = saved
        self.f_bin  = np.asarray(saved["f_bin"],           dtype=np.float64)
        self.Om_bin = np.asarray(saved["Omega_hat_bin"],   dtype=np.float64)
        self.sOm_bin = np.asarray(saved["sigma_Omega_bin"], dtype=np.float64)

        # Optimal-filter noise info (optional).
        self.P1_bin    = np.asarray(saved["P1_bin"],    dtype=np.float64) if "P1_bin"    in saved else None
        self.P2_bin    = np.asarray(saved["P2_bin"],    dtype=np.float64) if "P2_bin"    in saved else None
        self.gamma_bin = np.asarray(saved["gamma_bin"], dtype=np.float64) if "gamma_bin" in saved else None

        return saved

    # -------------------------
    # Injection-truth bookkeeping
    # -------------------------
    def set_lambda_true_values(self, true_values: Dict[str, float]) -> None:
        """
        Register the injection truth values for free Lambda parameters so
        downstream corner plots can draw them as reference markers.

        Only keys that match a name in lambda_param_specs are stored; the
        rest are silently ignored so the full Lambda dict can be passed.

        Example
        -------
        est.set_lambda_true_values({"alpha": 2.5, "beta": 1.0, "rate": 15.0})
        """
        valid = {s.name for s in self.lambda_param_specs}
        self.lambda_true_values = {k: float(v) for k, v in true_values.items() if k in valid}
