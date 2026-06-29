#!/usr/bin/env python3
# =============================================================================
# pipeline_simple_GR_CE_CE.py — Simplified GR-injection inference pipeline
# =============================================================================
#
# Single-stage pipeline that:
#   - Injects a pure-GR stochastic background (a = 0, alpha_ppE = 0)
#   - Runs nested-sampling inference with six physical parameters
#     plus k_sigma on a CE + CE network:
#         a, alpha_ppE, alpha_IMF, lambda_peak, rate, gamma  (+ k_sigma)
#   - Saves the posterior chain to disk
#
# `a` and `alpha_ppE` stay FREE even with GR injection so the run also
# tests whether the estimator recovers them as ~0.
#
# Compared to the production pipelines this version drops:
#   - the filename-based configuration parsing,
#   - the GR_MODE branch (a / alpha_ppE removed from inference),
#   - the analysis (Stage 2) and confidence-region (Stage 3) stages.
#
# All knobs live in the CONFIGURATION section below.
# =============================================================================

import os
import gc
import time
import datetime
import warnings
import logging

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import bilby

from SimulatedSignal import (
    SimulatedSignal,
    FrequencySettings,
    InjectionSettings,
    DetectorSettings,
    TimeDomainSettings,
    WelchSettings,
    PopulationSettings,
)

from ParametersEstimator import (
    DirectModelEvaluator,
    ParamGridSpec,
    estimate_C_pop,
)


# =============================================================================
# CONFIGURATION  (edit here)
# =============================================================================

# --- Detectors (CE + CE network) ---------------------------------------------
DET1 = "CE"
DET2 = "CE"

# --- Injection: pure GR (no ppE effect) --------------------------------------
A_TRUE         = 0.0
ALPHA_PPE_TRUE = 0.0

# --- Population truth (Lambda) -----------------------------------------------
LAMBDA_TRUE = {
    "alpha": 3.5, "beta": 1.0, "lam": 0.03, "mpp": 33.0, "sigpp": 4.5,
    "delta_m": 5.0, "rate": 17.0, "gamma": 3.0, "kappa": 5.7, "z_peak": 1.9,
    "mmin": 5.0, "mmax": 85.0,
}

# --- Free parameters and their priors ----------------------------------------
# `a` and `alpha_ppE` are kept free even with GR injection so the run also
# tests whether the estimator recovers them as ~0. The prior on alpha_ppE
# is therefore a small open interval bracketing zero.
A_MIN, A_MAX                 = 0.0, 5.0
ALPHA_PPE_MIN, ALPHA_PPE_MAX = 0.0, 0.10

FREE_LAMBDA_PARAMS = {
    "alpha": dict(min=0.0,  max=8.0,  latex=r"$\alpha_{\rm IMF}$"),
    "lam":   dict(min=0.0,  max=0.1,  latex=r"$\lambda_{\rm peak}$"),
    "rate":  dict(min=5.0,  max=40.0, latex=r"$\mathcal{R}\ [\mathrm{Gpc}^{-3}\mathrm{yr}^{-1}]$"),
    "gamma": dict(min=1.0,  max=7.0,  latex=r"$\gamma$"),
}

# --- Seeds -------------------------------------------------------------------
SEED_INJ = 1042
SEED_INF = 2042

# --- Dynesty sampler ---------------------------------------------------------
N_PROPOSAL_SAMPLES = 10000
NPOINTS = 1000
WALKS   = 5
DLOGZ   = 0.1

# --- Signal simulation -------------------------------------------------------
FFT_LENGTH  = 64
OVERLAP     = 32
PSD_SCALE   = 1
TD_DURATION = 512
TD_NSEGS    = 200
REBIN_DF_HZ = 1.0

# --- Intrinsic variance (sigma_rel) via population MC ------------------------
K_POP         = 20
POP_SEED_BASE = 90000

# --- Lambda cache (relaxed to maximize hits during rwalk) --------------------
LAMBDA_TOL_DECIMALS = 4
LRU_SIZE            = 32

# --- Parallelism -------------------------------------------------------------
N_CORES     = os.cpu_count() or 1
PHASE2_POOL = 8


# =============================================================================
# LIKELIHOOD — Lambda-dependent variance (Giarda et al. 2025, Eq. 19-20)
# =============================================================================
class OmegaLambdaVarianceLikelihood(bilby.Likelihood):
    """
    Gaussian likelihood with Lambda-dependent effective variance:

        sigma_eff^2(Lambda; f) =
            (k * sigma_inst(f))^2 + (sigma_rel(f) * Omega_model(Lambda; f))^2

    Reference: Giarda, Renzini, Pacilio, Gerosa (2025), arXiv:2506.12572
    """

    def __init__(self, f_bin, omega_hat, sigma_inst, sigma_rel,
                 evaluator, param_names):
        super().__init__(parameters={p: None for p in param_names + ["k_sigma"]})
        self.f          = np.asarray(f_bin,      dtype=np.float64)
        self.omega_hat  = np.asarray(omega_hat,  dtype=np.float64)
        self.sigma_inst = np.asarray(sigma_inst, dtype=np.float64)
        self.sigma_rel  = np.asarray(sigma_rel,  dtype=np.float64)
        self.evaluator  = evaluator
        self.param_names = list(param_names)
        self._log2pi    = float(np.log(2.0 * np.pi))

    def log_likelihood(self) -> float:
        k     = float(self.parameters["k_sigma"])
        theta = {p: float(self.parameters[p]) for p in self.param_names}
        m     = self.evaluator.model(theta)

        sigma_eff2 = (k * self.sigma_inst) ** 2 + (self.sigma_rel * m) ** 2
        valid = sigma_eff2 > 0.0
        if not np.any(valid):
            return -np.inf

        r     = self.omega_hat[valid] - m[valid]
        s2    = sigma_eff2[valid]
        log_l = -0.5 * np.sum(r ** 2 / s2 + np.log(s2) + self._log2pi)
        return float(log_l) if np.isfinite(log_l) else -np.inf


# =============================================================================
# HELPERS
# =============================================================================
def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _save_txt(path: str, header: str, **arrays) -> None:
    cols  = list(arrays.values())
    names = list(arrays.keys())
    data  = np.column_stack(cols)
    np.savetxt(path, data,
               header=header + "\n" + "  ".join(names),
               fmt="%.8e", comments="# ")


# =============================================================================
# PIPELINE
# =============================================================================
def main():
    t_total_start = time.time()

    _ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"outputs_simple_GR_{DET1}_{DET2}_{_ts}"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print(f"[Pipeline] Simple GR-injection test  ({DET1} + {DET2})")
    print("=" * 60)
    print(f"  Injection: a={A_TRUE}  alpha_ppE={ALPHA_PPE_TRUE}")
    print(f"  N_CORES={N_CORES}  pool={PHASE2_POOL}")
    print(f"  NPOINTS={NPOINTS}  WALKS={WALKS}  DLOGZ={DLOGZ}")
    print(f"  N_PROPOSAL={N_PROPOSAL_SAMPLES}  K_POP={K_POP}")
    print(f"  Outdir: {outdir}")
    print()

    # --- Persist a tiny run_config.txt for reproducibility -------------------
    cfg_path = os.path.join(outdir, "run_config.txt")
    with open(cfg_path, "w") as f:
        f.write(f"A_TRUE            = {A_TRUE}\n")
        f.write(f"ALPHA_PPE_TRUE    = {ALPHA_PPE_TRUE}\n")
        f.write(f"DETECTORS         = {DET1}_{DET2}\n")
        f.write(f"N_PROPOSAL        = {N_PROPOSAL_SAMPLES}\n")
        f.write(f"NPOINTS           = {NPOINTS}\n")
        f.write(f"WALKS             = {WALKS}\n")
        f.write(f"DLOGZ             = {DLOGZ}\n")
        f.write(f"K_POP             = {K_POP}\n")
        f.write(f"LAMBDA_TOL_DEC    = {LAMBDA_TOL_DECIMALS}\n")
        f.write(f"LRU_SIZE          = {LRU_SIZE}\n")
        f.write(f"SEED_INJ          = {SEED_INJ}\n")
        f.write(f"SEED_INF          = {SEED_INF}\n")
        f.write(f"RUN_TIMESTAMP     = {_ts}\n")

    # -------------------------------------------------------------------------
    # PHASE 0a — Build the population and inject the GR signal
    # -------------------------------------------------------------------------
    freq = FrequencySettings(
        fmin=10.0, fmax=250.0, n_freq_eff=400,
        n_proposal_samples=N_PROPOSAL_SAMPLES,
        rebin_df_hz=REBIN_DF_HZ, gamma_min=0.05, floor=1e-60,
    )
    det = DetectorSettings(
        det1_name=DET1, det2_name=DET2,
        det3_name=None, det4_name=None,
        placeholder_if_missing=False, psd_scale=PSD_SCALE,
    )
    td = TimeDomainSettings(
        duration=TD_DURATION, n_segs=TD_NSEGS, fs=1024,
        seed_noise=123, seed_signal=int(SEED_INJ),
    )
    welch = WelchSettings(fft_length=FFT_LENGTH, overlap=OVERLAP)
    pop   = PopulationSettings(
        binary_type="BBH", waveform_approximant="IMRPhenomD",
        inspiral_only=True, disable_inspiral_cutoff=True,
        minimum_frequency=freq.fmin, sampling_frequency=4096.0, duration=4.0,
        fast_binning=True, max_bins=100, use_frequency_warp=True, chunksize=50,
    )

    print("[Phase 0a] Injecting GR signal (a=0, alpha_ppE=0)...", flush=True)
    t0 = time.time()
    inj = InjectionSettings(a_true=A_TRUE, alpha_true=ALPHA_PPE_TRUE)
    sim = SimulatedSignal(
        freq=freq, inj=inj, det=det, td=td, welch=welch,
        popset=pop, Lambda=dict(LAMBDA_TRUE),
    )
    np.random.seed(int(SEED_INJ))
    sim.precompute_population()
    omega_inj, _extras_inj = sim.build_injected_omega(
        a=A_TRUE, alpha_ppE=ALPHA_PPE_TRUE,
    )
    print(f"[Phase 0a] Done in {_fmt_elapsed(time.time() - t0)}\n", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 0b — Analytical Omega_hat on the final rebinned grid
    # -------------------------------------------------------------------------
    print("[Phase 0b] Running simulate_analytical...", flush=True)
    t0b = time.time()
    f_raw, Om_raw, sOm_raw, f_bin, Om_bin, sOm_bin, \
        P1_bin, P2_bin, gamma_bin, meta = sim.simulate_analytical(omega_inj)
    sOm_bin_inst = np.asarray(sOm_bin, dtype=float)
    print(f"[Phase 0b] Done in {_fmt_elapsed(time.time() - t0b)}   "
          f"N_bins={len(f_bin)}\n", flush=True)

    _save_txt(
        os.path.join(outdir, "omega_bin.txt"),
        header=f"GR injection (a={A_TRUE}, alpha_ppE={ALPHA_PPE_TRUE})  "
               f"rebin_df={REBIN_DF_HZ}Hz",
        f_hz=f_bin, omega_hat=Om_bin, sigma_omega=sOm_bin_inst,
    )

    # Reusable population objects (used both by sigma_rel and the evaluator)
    _freqs_pop     = np.asarray(sim._ctx.frequencies, dtype=float)
    _omega_fid     = np.asarray(sim._omega_fid,       dtype=float)
    _dataset       = sim._dataset
    _ctx           = sim._ctx
    _probabilities = np.asarray(sim._probabilities,   dtype=float)

    # -------------------------------------------------------------------------
    # PHASE 1 — Intrinsic variance via population MC (sigma_rel)
    # -------------------------------------------------------------------------
    phase1_jobs = min(K_POP, N_CORES)
    print(f"[Phase 1] Estimating sigma_rel with K_POP={K_POP} realisations "
          f"(n_jobs={phase1_jobs})...", flush=True)
    t1 = time.time()

    _C_pop_unused, omega_pop_samples = estimate_C_pop(
        f_bin=f_bin,
        a_ref=A_TRUE,
        alpha_ref_scalar=ALPHA_PPE_TRUE,
        freq_settings=freq, det_settings=det, td_settings=td,
        welch_settings=welch, pop_settings=pop,
        k_pop=K_POP, seed_base=POP_SEED_BASE,
        jitter=1e-10, n_jobs=phase1_jobs,
        Lambda=LAMBDA_TRUE,
    )
    omega_pop_samples = np.asarray(omega_pop_samples, dtype=float)

    _mean_pop = np.mean(omega_pop_samples, axis=0)
    _std_pop  = np.std( omega_pop_samples, axis=0, ddof=1)
    sigma_rel = np.where(_mean_pop > 0,
                         _std_pop / np.maximum(_mean_pop, 1e-300), 0.0)
    sigma_rel = np.clip(sigma_rel, 1e-6, 1.0)

    del omega_pop_samples, _C_pop_unused
    gc.collect()
    print(f"[Phase 1] Done in {_fmt_elapsed(time.time() - t1)}   "
          f"median sigma_rel = {np.median(sigma_rel):.4f}\n", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 2 — Build the evaluator + likelihood + priors
    # -------------------------------------------------------------------------
    # Parameter specs in the order they appear in the posterior chain.
    param_specs = [
        ParamGridSpec(name="a",         min=A_MIN,         max=A_MAX,
                      n=1, transform="linear", latex_label=r"$a$"),
        ParamGridSpec(name="alpha_ppE", min=ALPHA_PPE_MIN, max=ALPHA_PPE_MAX,
                      n=1, transform="linear", latex_label=r"$\alpha_{\rm ppE}$"),
    ] + [
        ParamGridSpec(name=name, min=v["min"], max=v["max"],
                      n=1, transform="linear", latex_label=v["latex"])
        for name, v in FREE_LAMBDA_PARAMS.items()
    ]
    param_names        = [s.name for s in param_specs]
    lambda_param_names = list(FREE_LAMBDA_PARAMS.keys())

    # base_Lambda fixes all hyperparameters not explicitly varied.
    base_lambda_run = {**LAMBDA_TRUE, "a": A_TRUE, "alpha_ppE": ALPHA_PPE_TRUE}

    evaluator = DirectModelEvaluator(
        f_bin=np.asarray(f_bin, dtype=np.float64),
        freqs_pop=_freqs_pop, omega_fid=_omega_fid,
        dataset=_dataset, base_Lambda=base_lambda_run,
        ctx=_ctx, probabilities=_probabilities,
        lambda_param_names=lambda_param_names,
        alpha_param_names=[],   # scalar alpha_ppE mode
        G_param_names=[],       # no G correction
        alpha_event_model=None,
        G_event_weight=None,
        compute_knobs=dict(nproc=1, fast_binning=True, max_bins=100,
                           use_frequency_warp=True, chunksize=50,
                           disable_inspiral_cutoff=True),
        pop_cfg=dict(
            FMIN=float(freq.fmin), FMAX=float(freq.fmax),
            N_FREQ=int(freq.n_freq_eff), N_SYS=N_PROPOSAL_SAMPLES,
            binary_type=str(pop.binary_type),
            waveform_approximant=str(pop.waveform_approximant),
            inspiral_only=bool(pop.inspiral_only),
            disable_inspiral_cutoff=bool(pop.disable_inspiral_cutoff),
            sampling_frequency=float(pop.sampling_frequency),
            duration=float(pop.duration),
        ),
        lambda_tol_decimals=LAMBDA_TOL_DECIMALS,
        lru_size=LRU_SIZE,
    )

    likelihood = OmegaLambdaVarianceLikelihood(
        f_bin=f_bin, omega_hat=Om_bin, sigma_inst=sOm_bin_inst,
        sigma_rel=sigma_rel, evaluator=evaluator, param_names=param_names,
    )

    priors = bilby.core.prior.PriorDict()
    for s in param_specs:
        priors[s.name] = bilby.core.prior.Uniform(
            float(s.min), float(s.max),
            name=s.name, latex_label=s.latex_label)
    priors["k_sigma"] = bilby.core.prior.Uniform(
        minimum=0.5, maximum=2.0, name="k_sigma", latex_label=r"$k_\sigma$")

    # -------------------------------------------------------------------------
    # PHASE 3 — Nested sampling (dynesty)
    # -------------------------------------------------------------------------
    print(f"[Phase 3] Running dynesty (npoints={NPOINTS}, walks={WALKS}, "
          f"dlogz={DLOGZ}, npool={PHASE2_POOL})...", flush=True)
    t3 = time.time()
    np.random.seed(int(SEED_INF))

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=NPOINTS,
        walks=WALKS,
        dlogz=DLOGZ,
        npool=PHASE2_POOL,
        outdir=outdir,
        label="infer_simple_GR",
        check_point=False,
        resume=False,
        clean=True,
        check_point_plot=False,
        sample="rwalk",
        bound="multi",
        proposals=["diff", "volumetric"],
        update_interval=600,
        queue_size=24,
    )
    elapsed3 = time.time() - t3
    print(f"[Phase 3] Done in {_fmt_elapsed(elapsed3)}\n", flush=True)
    gc.collect()

    # -------------------------------------------------------------------------
    # PHASE 4 — Save the posterior chain
    # -------------------------------------------------------------------------
    post = result.posterior
    chain_path = os.path.join(outdir, "mcmc_chain.txt")
    header = (
        f"MCMC posterior chain — simple GR-injection test\n"
        f"# injection: a={A_TRUE}, alpha_ppE={ALPHA_PPE_TRUE}\n"
        f"# N_samples={len(post)}\n"
        f"# {'  '.join(post.columns)}"
    )
    np.savetxt(chain_path, post.values, header=header, fmt="%.8e", comments="# ")
    print(f"[Phase 4] Chain saved: {chain_path}   ({len(post)} samples)\n",
          flush=True)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Posterior medians:")
    truth_map = {"a": A_TRUE, "alpha_ppE": ALPHA_PPE_TRUE,
                 **LAMBDA_TRUE, "k_sigma": 1.0}
    for p in post.columns:
        med = float(np.median(post[p].values))
        std = float(np.std(post[p].values))
        true_val = truth_map.get(p)
        tstr = f"  (true={true_val:.4g})" if true_val is not None else ""
        print(f"  {p:12s}: {med:.4g} +/- {std:.4g}{tstr}")
    print("=" * 60)

    t_total = time.time() - t_total_start
    print(f"Total wall time: {_fmt_elapsed(t_total)}")

    # Append timings to run_config.txt
    with open(cfg_path, "a") as f:
        f.write(f"\n# Timings\n")
        f.write(f"  Phase 3 (dynesty) : {_fmt_elapsed(elapsed3)}\n")
        f.write(f"  TOTAL             : {_fmt_elapsed(t_total)}\n")
        try:
            f.write(f"\n# Evidence\n")
            f.write(f"  log_evidence     = {result.log_evidence:.4f}\n")
            f.write(f"  log_evidence_err = {result.log_evidence_err:.4f}\n")
        except AttributeError:
            pass


if __name__ == "__main__":
    main()
