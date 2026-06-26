#!/usr/bin/env python3
"""
Example: waveform-level correction of Omega_GW for a BBH population
===================================================================
Generates 4 independent figures, each with 2 subplots:
  top subplot    : Omega_GW(f) — baseline and with the correction
  bottom subplot : relative difference (Omega_corr - Omega_base) / Omega_base

  Fig 1 — lines: varying 'a',     alpha fixed
  Fig 2 — lines: varying 'alpha', a fixed
  Fig 3 — fill:  range of 'a',     alpha fixed
  Fig 4 — fill:  range of 'alpha', a fixed

The expensive part (waveform / h2_cache) is computed ONCE and reused.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inspect

import CorrectionOmegaGW as corr
from popstock.PopulationOmegaGW import PopulationOmegaGW
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.grid'] = False          # global grid disabled
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 24

# ============================================================
# CONFIGURATION
# ============================================================

FMIN = 10.0
FMAX = 2048.0
N_FREQ_EFF    = 400
N_PROPOSAL    = 10000    # increase for a smoother Monte Carlo (cost: more waveform calls)
SEED          = 1234

# -- Reference values -----------------------------------------
A_FIX     = 4.0          # a fixed (used when varying alpha)
ALPHA_FIX = 0.1          # alpha fixed (used when varying a)

# -- Discrete values for the line plots -----------------------
A_DISCRETE     = [-8.0, -2.0, 4.0]
ALPHA_DISCRETE = [0.1, 0.5, 1.0]

# -- Dense ranges for the fill plots --------------------------
A_FILL     = np.linspace(0.5, 6.0, 30)
ALPHA_FILL = np.logspace(0, 4, 30)

# -- BBH population hyperparameters ---------------------------
Lambda_BBH = {
    "alpha": 2.5, "beta": 1.0, "delta_m": 3.0, "lam": 0.04,
    "mmin": 5.0,  "mmax": 80.0, "mpp": 33.0,  "sigpp": 5.0,
    "gamma": 2.7, "kappa": 5.0, "z_peak": 1.9,
    "rate": 20.0,
}


# ============================================================
# HELPERS — PopStock compatibility
# ============================================================

def _infer_required_kwargs(callable_obj):
    """Return the required keyword-argument names of a model callable,
    excluding 'self' and the leading dataset argument."""
    try:
        sig = inspect.signature(callable_obj)
    except TypeError:
        sig = inspect.signature(callable_obj.__call__)
    params = list(sig.parameters.values())
    nonself = [p for p in params if p.name != "self"]
    if nonself:
        nonself = nonself[1:]   # drop the dataset (first real argument)
    return [p.name for p in nonself
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def _ensure_popstock_args_and_fiducial(pop, Lambda):
    """Populate the model_args / fiducial-parameter attributes expected by
    popstock, inferring the per-model argument names when needed."""
    if not hasattr(pop, "model_args") or pop.model_args is None:
        pop.model_args = {}
    mass_keys = _infer_required_kwargs(pop.models["mass"]) or [
        "alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "beta", "delta_m"
    ]
    red_keys  = _infer_required_kwargs(pop.models["redshift"]) or [
        "gamma", "kappa", "z_peak"
    ]
    pop.model_args["mass"]     = mass_keys
    pop.model_args["redshift"] = red_keys
    fid = {k: Lambda[k] for k in mass_keys + red_keys + ["rate"] if k in Lambda}
    for attr in ("fiducial_parameters", "fiducial_params", "_fiducial_parameters"):
        if hasattr(pop, attr):
            try:
                setattr(pop, attr, fid)
            except Exception:
                pass
    return fid


def pop_draw_samples(pop, Lambda, N, seed):
    """Draw proposal samples, trying the available backend signatures."""
    np.random.seed(int(seed))
    fid = _ensure_popstock_args_and_fiducial(pop, Lambda)
    sig = inspect.signature(pop.draw_and_set_proposal_samples)
    for method in ("direct", "grid"):
        try:
            kw = dict(N_proposal_samples=int(N), mass=method, redshift=method)
            if "seed" in sig.parameters:
                kw["seed"] = int(seed)
            return pop.draw_and_set_proposal_samples(fid, **kw)
        except (UnboundLocalError, TypeError):
            try:
                kw = dict(N_proposal_samples=int(N))
                if "seed" in sig.parameters:
                    kw["seed"] = int(seed)
                return pop.draw_and_set_proposal_samples(fid, **kw)
            except Exception:
                continue
    raise RuntimeError("Failed to draw proposal samples.")


def pop_calculate_omega(pop, Lambda, approximant, fs, fmin, multiprocess=False):
    """Call the backend Omega_GW computation, forwarding only the keyword
    arguments supported by the installed version."""
    sig = inspect.signature(pop.calculate_omega_gw)
    kw  = dict(
        waveform_approximant=approximant,
        sampling_frequency=fs,
        waveform_minimum_frequency=fmin,
        minimum_frequency=fmin,
        multiprocess=multiprocess,
    )
    if "Lambda" in sig.parameters:
        supported = {k: v for k, v in kw.items() if k in sig.parameters}
        return pop.calculate_omega_gw(Lambda=Lambda, **supported)
    supported = {k: v for k, v in kw.items() if k in sig.parameters}
    return pop.calculate_omega_gw(Lambda, **supported)


def G_unity(dataset):
    """Per-event weight equal to 1 for every event (no correction)."""
    return np.ones_like(dataset["redshift"], dtype=float)


# ============================================================
# ONE-TIME POPULATION SETUP
# ============================================================

def setup_population():
    """
    Build the context, popstock object, dataset, probabilities and h2_cache.
    Everything that depends on (a, alpha) is deferred to later.
    """
    print("[setup] Building models and context...")
    models = {
        "mass":     SinglePeakSmoothedMassDistribution(mmin=2.0, mmax=100.0),
        "redshift": MadauDickinsonRedshift(z_max=10.0),
    }
    wf_cfg = corr.WaveformConfig(
        binary_type="BBH",
        waveform_approximant="IMRPhenomD",
        inspiral_only=True,
        disable_inspiral_cutoff=True,
        minimum_frequency=FMIN,
        sampling_frequency=4096.0,
        duration=4.0,
    )
    fgrid_cfg = corr.FreqGridConfig(fmin=FMIN, fmax=FMAX, N_freq_eff=N_FREQ_EFF)

    wg = corr.build_waveform_generator(wf_cfg)
    frequencies, idx = corr.build_frequency_grid(wg.frequency_array, fgrid_cfg)

    ctx = corr.CorrectionContext(
        models=models, wf_cfg=wf_cfg, fgrid_cfg=fgrid_cfg,
        frequencies=frequencies, idx_freq=idx,
    )

    # -- PopStock ---------------------------------------------
    print(f"[setup] Drawing {N_PROPOSAL} proposal samples...")
    pop = PopulationOmegaGW(models=models, frequency_array=frequencies)
    pop_draw_samples(pop, Lambda_BBH, N_PROPOSAL, SEED)
    pop_calculate_omega(pop, Lambda_BBH, "IMRPhenomD", 4096.0, FMIN)

    dataset  = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)
    omega_gr = np.asarray(pop.omega_gw, dtype=float)
    omega_gr = np.where(np.isfinite(omega_gr) & (omega_gr >= 0), omega_gr, 0.0)

    # -- Probabilities ----------------------------------------
    pop_tmp = PopulationOmegaGW(models=models)
    _ensure_popstock_args_and_fiducial(pop_tmp, Lambda_BBH)
    probabilities = np.asarray(
        pop_tmp.calculate_probabilities(dataset, Lambda_BBH), dtype=float
    )

    # -- h2_cache: the expensive part, independent of (a, alpha)
    print("[setup] Precomputing h2_cache (waveform calls)...")
    h2_cache, f_peak_obs = corr.compute_h2_cache_and_fpeak_parallel(
        dataset=dataset, idx=idx, wf_cfg=wf_cfg,
        frequencies=frequencies, nproc=None, chunksize=20,
        fast_binning=True, max_bins=250, use_frequency_warp=True,
    )
    print("[setup] Done.\n")

    return ctx, dataset, omega_gr, probabilities, h2_cache, f_peak_obs, frequencies


# ============================================================
# CORRECTION EVALUATION (fast — uses the precomputed h2_cache)
# ============================================================

def eval_correction(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                    omega_gr, a, alpha):
    """Return omega_corr for a single (a, alpha) pair using the cached arrays."""
    corr_ppE, corr_G, _ = corr.compute_correction(
        dataset=dataset,
        Lambda=Lambda_BBH,
        ctx=ctx,
        a=float(a),
        alpha_ppE=float(alpha),
        G_event_weight=G_unity,
        precomputed_probabilities=probabilities,
        precomputed_h2_cache=h2_cache,
        precomputed_f_peak_obs=f_peak_obs,
        nproc=1,
        chunksize=20,
        fast_binning=True,
        max_bins=250,
        use_frequency_warp=True,
        disable_inspiral_cutoff=True,
    )
    oc = omega_gr * np.asarray(corr_ppE, float) * np.asarray(corr_G, float)
    return np.where(np.isfinite(oc) & (oc >= 0), oc, 0.0)


# ============================================================
# FIGURE HELPERS
# ============================================================

def _make_fig():
    """
    Create a figure with two subplots sharing the x-axis:
      top (3/4): Omega_GW spectrum
      bottom (1/4): relative difference
    """
    fig, (ax_s, ax_r) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.05},
    )
    return fig, ax_s, ax_r


def _style_fig(fig, ax_s, ax_r, ylim_rel=None):
    """Apply shared axis scales, labels and styling to both subplots."""
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_ylabel(r"$\Omega_{\rm GW}(f)$", fontsize=24)
    ax_s.legend(fontsize=16, framealpha=0.9)
    ax_s.grid(False)

    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$f\ [\mathrm{Hz}]$", fontsize=24)
    ax_r.set_ylabel(r"$\frac{\Delta\Omega}{\Omega_{\rm GR}}$", fontsize=24)
    ax_r.axhline(0, color="k", lw=1.2, ls="--", zorder=5)
    ax_r.grid(False)
    if ylim_rel is not None:
        ax_r.set_ylim(*ylim_rel)

    plt.tight_layout()


def _safe_band(arr_2d):
    """Column-wise min/max ignoring zeros (which would break the log scale)."""
    safe = np.where(arr_2d > 0, arr_2d, np.nan)
    lo = np.nanmin(safe, axis=0)
    hi = np.nanmax(safe, axis=0)
    return lo, hi


# ============================================================
# FIG 1 — LINES, varying a, alpha fixed
# ============================================================

def fig1_lines_vary_a(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                      omega_gr, freqs):
    """Plot the spectrum and relative difference for discrete values of 'a'."""
    print(f"[Fig 1] Computing {len(A_DISCRETE)} corrections (varying a)...")
    band = np.isfinite(omega_gr) & (omega_gr > 0)
    f, og = freqs[band], omega_gr[band]

    fig, ax_s, ax_r = _make_fig()

    # Baseline
    ax_s.loglog(f, og, "k-", lw=2.5, zorder=10, label="GR")

    for a_val in A_DISCRETE:
        oc  = eval_correction(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                              omega_gr, a_val, ALPHA_FIX)[band]
        rel = (oc - og) / og
        ax_s.loglog(f, oc, lw=1.8,
                    label=rf"$a = {a_val:.0f}$")
        ax_r.loglog(f, rel, lw=1.8)

    _style_fig(fig, ax_s, ax_r)
    fig.savefig("fig1_lines_vary_a.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_lines_vary_a.pdf")


# ============================================================
# FIG 2 — LINES, varying alpha, a fixed
# ============================================================

def fig2_lines_vary_alpha(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                          omega_gr, freqs):
    """Plot the spectrum and relative difference for discrete values of 'alpha'."""
    print(f"[Fig 2] Computing {len(ALPHA_DISCRETE)} corrections (varying alpha)...")
    band = np.isfinite(omega_gr) & (omega_gr > 0)
    f, og = freqs[band], omega_gr[band]

    fig, ax_s, ax_r = _make_fig()
    ax_s.loglog(f, og, "k-", lw=2.5, zorder=10, label="GR")

    for alpha_val in ALPHA_DISCRETE:
        oc  = eval_correction(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                              omega_gr, A_FIX, alpha_val)[band]
        rel = (oc - og) / og
        ax_s.loglog(f, oc, lw=1.8,
                    label=rf"$\alpha = {alpha_val:.2f}$")
        ax_r.loglog(f, rel, lw=1.8)

    _style_fig(fig, ax_s, ax_r)
    fig.savefig("fig2_lines_vary_alpha.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_lines_vary_alpha.pdf")


# ============================================================
# FIG 3 — FILL, range of a, alpha fixed
# ============================================================

def fig3_fill_vary_a(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                     omega_gr, freqs):
    """Plot a shaded band spanning a range of 'a', with the central curve."""
    n = len(A_FILL)
    print(f"[Fig 3] Computing {n} corrections for the band (varying a)...")
    band = np.isfinite(omega_gr) & (omega_gr > 0)
    f, og = freqs[band], omega_gr[band]

    stack = np.array([
        eval_correction(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                        omega_gr, a_val, ALPHA_FIX)[band]
        for a_val in A_FILL
    ])
    lo, hi = _safe_band(stack)
    mid    = stack[n // 2]
    rel_lo = (lo - og) / og
    rel_hi = (hi - og) / og
    rel_mi = (mid - og) / og

    COLOR = "#2E86AB"

    fig, ax_s, ax_r = _make_fig()

    ax_s.loglog(f, og, "k-", lw=2.5, zorder=10, label="GR")
    ax_s.fill_between(f, lo, hi, color=COLOR, alpha=0.30, zorder=2,
                      label=(rf"$a \in [{A_FILL[0]:.1f},\,"
                             rf"{A_FILL[-1]:.1f}]$,"
                             rf"\ $\alpha={ALPHA_FIX:.0e}$"))
    ax_s.loglog(f, mid, color=COLOR, lw=2.0, ls="--", zorder=3,
                label=rf"$a = {A_FILL[n//2]:.1f}$ (central)")

    ax_r.fill_between(f, rel_lo, rel_hi, color=COLOR, alpha=0.30)
    ax_r.semilogx(f, rel_mi, color=COLOR, lw=2.0, ls="--")

    _style_fig(fig, ax_s, ax_r)
    fig.savefig("fig3_fill_vary_a.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_fill_vary_a.pdf")


# ============================================================
# FIG 4 — FILL, range of alpha, a fixed
# ============================================================

def fig4_fill_vary_alpha(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                         omega_gr, freqs):
    """Plot a shaded band spanning a range of 'alpha', with the central curve."""
    n = len(ALPHA_FILL)
    print(f"[Fig 4] Computing {n} corrections for the band (varying alpha)...")
    band = np.isfinite(omega_gr) & (omega_gr > 0)
    f, og = freqs[band], omega_gr[band]

    stack = np.array([
        eval_correction(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                        omega_gr, A_FIX, alpha_val)[band]
        for alpha_val in ALPHA_FILL
    ])
    lo, hi = _safe_band(stack)
    mid    = stack[n // 2]
    rel_lo = (lo - og) / og
    rel_hi = (hi - og) / og
    rel_mi = (mid - og) / og

    COLOR = "#E84855"

    fig, ax_s, ax_r = _make_fig()

    ax_s.loglog(f, og, "k-", lw=2.5, zorder=10, label="GR")
    ax_s.fill_between(f, lo, hi, color=COLOR, alpha=0.30, zorder=2,
                      label=(rf"$\alpha \in [10^{{{int(np.log10(ALPHA_FILL[0]))}}},\,"
                             rf"10^{{{int(np.log10(ALPHA_FILL[-1]))}}}]$,"
                             rf"\ $a={A_FIX:.0f}$"))
    ax_s.loglog(f, mid, color=COLOR, lw=2.0, ls="--", zorder=3,
                label=(rf"$\alpha = 10^{{{np.log10(ALPHA_FILL[n//2]):.1f}}}$ (central)"))

    ax_r.fill_between(f, rel_lo, rel_hi, color=COLOR, alpha=0.30)
    ax_r.semilogx(f, rel_mi, color=COLOR, lw=2.0, ls="--")

    _style_fig(fig, ax_s, ax_r)
    fig.savefig("fig4_fill_vary_alpha.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_fill_vary_alpha.pdf")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the one-time setup and generate all four figures."""
    ctx, dataset, omega_gr, probabilities, h2_cache, f_peak_obs, freqs = \
        setup_population()

    fig1_lines_vary_a(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                      omega_gr, freqs)
    fig2_lines_vary_alpha(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                          omega_gr, freqs)
    fig3_fill_vary_a(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                     omega_gr, freqs)
    fig4_fill_vary_alpha(ctx, dataset, probabilities, h2_cache, f_peak_obs,
                         omega_gr, freqs)

    print("\nDone. Figures generated!")

if __name__ == "__main__":
    main()
