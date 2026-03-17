

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette (accessible, print-friendly)
COL_IPCW = "#2166AC"      # deep blue
COL_COMP = "#D6604D"      # muted red
COL_DIFF = "#4D4D4D"      # dark grey
COL_HIST = "#4393C3"      # steel blue
COL_ZERO = "#999999"      # light grey for reference lines


# -------------
# Kaplan-Meier 

def kaplan_meier_survival(times, events):

    n = len(times)
    surv = np.ones(n)
    surv_minus = np.ones(n)
    cum = 1.0
    for j in range(n):
        surv_minus[j] = cum
        rj = n - j
        cum *= (1.0 - events[j] / rj)
        surv[j] = cum
    return surv, surv_minus


def km_jump_masses(events, surv_minus):
    """
    Compute KM jump masses: p_j = Delta_j * S_hat(t_j^-) / R_j.
    """
    n = len(events)
    rj = np.arange(n, 0, -1, dtype=float)
    return events * surv_minus / rj


# ---------------------------
# Score functions for the exponential model

def exp_score(t, theta):
    """
    Score phi(t; theta) = d/d(theta) log f_theta(t) for the exponential
    model f_theta(t) = exp(theta) * exp(-exp(theta)*t).

    phi(t; theta) = 1 - exp(theta)*t.
    """
    return 1.0 - np.exp(theta) * t


def compute_U_comp(Y_ord, Delta_ord, theta):
    """
    Completion-induced score U_comp(theta).
    """
    n = len(Y_ord)
    surv, surv_minus = kaplan_meier_survival(Y_ord, Delta_ord)
    pj = km_jump_masses(Delta_ord, surv_minus)

    score_vals = exp_score(Y_ord, theta)
    total = 0.0

    for i in range(n):
        if Delta_ord[i] == 1:
            total += score_vals[i]
        else:
            # Censored: complete from the conditional KM tail
            s_yi = surv[i]  # S_hat(Y_(i)), right-continuous = S_hat after step
            # But since Y_(i) is a censoring time, S_hat doesn't jump here,
            # so S_hat(Y_(i)) = S_hat(Y_(i)^-) * (1 - 0/R_i) = S_hat(Y_(i)^-)
            # i.e., surv[i] is correct (no event jump at this index).
            if s_yi > 0:
                for j in range(i + 1, n):
                    if Delta_ord[j] == 1:
                        total += (pj[j] / s_yi) * score_vals[j]
    return total


def compute_U_ipcw(Y_ord, Delta_ord, theta):
    """
    IPCW score U_IPCW(theta) = sum_i delta_i / G_hat(Y_i^-) * phi(Y_i; theta).
    """
    n = len(Y_ord)
    # Censoring KM: swap events and censorings
    cens_indicators = 1.0 - Delta_ord
    _, G_minus = kaplan_meier_survival(Y_ord, cens_indicators)

    score_vals = exp_score(Y_ord, theta)
    total = 0.0
    for i in range(n):
        if Delta_ord[i] == 1:
            total += score_vals[i] / G_minus[i]
    return total


# --------
# Data generation

def generate_censored_data(n, theta0, lam_c, rng):
    """
    Generate right-censored data from Exp(exp(theta0)) with censoring Exp(lam_c).
    Returns ordered (Y, Delta) with distinct times.
    """
    rate = np.exp(theta0)
    T = rng.exponential(1.0 / rate, size=n)
    C = rng.exponential(1.0 / lam_c, size=n)
    Y = np.minimum(T, C)
    delta = (T <= C).astype(float)

    # Order by Y
    idx = np.argsort(Y)
    Y_ord = Y[idx]
    Delta_ord = delta[idx]
    return Y_ord, Delta_ord


def check_assumptions(Y_ord, Delta_ord):
    """Check: distinct times and largest observation is an event."""
    distinct = len(np.unique(Y_ord)) == len(Y_ord)
    last_event = Delta_ord[-1] == 1
    return distinct and last_event


def find_root(score_func, Y_ord, Delta_ord, bracket=(-3, 3)):
    """Find the root of a score function using Brent's method."""
    try:
        root = brentq(lambda th: score_func(Y_ord, Delta_ord, th),
                       bracket[0], bracket[1], xtol=1e-12, rtol=1e-14)
        return root
    except ValueError:
        return np.nan


# ----------
# Figure 1: Score curves and difference (two-panel)


def make_figure1(Y_ord, Delta_ord, seed_label, outdir="."):
    """Create a two-panel figure: scores overlaid + difference."""
    theta_grid = np.linspace(-2.0, 1.5, 500)

    U_comp_vals = np.array([compute_U_comp(Y_ord, Delta_ord, th) for th in theta_grid])
    U_ipcw_vals = np.array([compute_U_ipcw(Y_ord, Delta_ord, th) for th in theta_grid])
    diff_vals = U_comp_vals - U_ipcw_vals

    # Roots
    root_comp = find_root(compute_U_comp, Y_ord, Delta_ord)
    root_ipcw = find_root(compute_U_ipcw, Y_ord, Delta_ord)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # --- Left panel: overlaid scores ---
    ax1.plot(theta_grid, U_ipcw_vals, color=COL_IPCW, linewidth=2.0,
             label=r"$U_{\mathrm{IPCW}}(\theta)$", zorder=3)
    ax1.plot(theta_grid, U_comp_vals, color=COL_COMP, linewidth=1.2,
             linestyle="--", dashes=(5, 3),
             label=r"$U_{\mathrm{comp}}(\theta)$", zorder=4)
    ax1.axhline(0, color=COL_ZERO, linewidth=0.6, linestyle="-", zorder=1)

    # Mark roots
    ax1.plot(root_ipcw, 0, "o", color=COL_IPCW, markersize=5, zorder=5)
    ax1.plot(root_comp, 0, "s", color=COL_COMP, markersize=5, zorder=5,
             markerfacecolor="none", markeredgewidth=1.5)

    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("Score")
    ax1.set_title("(a) Completion and IPCW scores", fontsize=11, fontweight="normal")
    ax1.legend(frameon=False, loc="upper right")

    # --- Right panel: difference ---
    ax2.plot(theta_grid, diff_vals, color=COL_DIFF, linewidth=1.0)
    ax2.axhline(0, color=COL_ZERO, linewidth=0.6, linestyle="-", zorder=1)

    max_diff = np.max(np.abs(diff_vals))
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$U_{\mathrm{comp}}(\theta) - U_{\mathrm{IPCW}}(\theta)$")
    ax2.set_title("(b) Score difference", fontsize=11, fontweight="normal")

    # Annotate max difference
    ax2.annotate(
        f"max $|\\Delta(\\theta)|$ = {max_diff:.1e}",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    # Set y-axis limits to show scale of numerical noise
    ylim = max(max_diff * 2.5, 1e-15)
    ax2.set_ylim(-ylim, ylim)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))

    fig.tight_layout(w_pad=3.0)

    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig1_scores.{ext}")
    plt.close(fig)

    print(f"  Figure 1 saved. max|Delta(theta)| = {max_diff:.2e}, "
          f"root_comp = {root_comp:.8f}, root_ipcw = {root_ipcw:.8f}, "
          f"|root diff| = {abs(root_comp - root_ipcw):.2e}")

    return max_diff, abs(root_comp - root_ipcw)


# --------------------------------------
# Figure 2: Monte Carlo histogram of root differences

def make_figure2(n, theta0, lam_c, n_mc, outdir="."):
    """
    Run Monte Carlo replications and plot a histogram of
    |theta_hat_comp - theta_hat_IPCW|.
    """
    rng = np.random.default_rng(2026)
    root_diffs = []
    max_score_diffs = []
    n_valid = 0

    for rep in range(n_mc * 3):  # generate extras in case some don't meet assumptions
        if n_valid >= n_mc:
            break
        Y_ord, Delta_ord = generate_censored_data(n, theta0, lam_c, rng)
        if not check_assumptions(Y_ord, Delta_ord):
            continue
        n_valid += 1

        root_comp = find_root(compute_U_comp, Y_ord, Delta_ord)
        root_ipcw = find_root(compute_U_ipcw, Y_ord, Delta_ord)
        if np.isnan(root_comp) or np.isnan(root_ipcw):
            continue
        root_diffs.append(abs(root_comp - root_ipcw))

        # Also check score difference at a single point for summary
        diff_at_0 = abs(compute_U_comp(Y_ord, Delta_ord, 0.0)
                        - compute_U_ipcw(Y_ord, Delta_ord, 0.0))
        max_score_diffs.append(diff_at_0)

    root_diffs = np.array(root_diffs)
    max_score_diffs = np.array(max_score_diffs)

    print(f"  Monte Carlo: {len(root_diffs)} valid replications")
    print(f"  Root differences: median = {np.median(root_diffs):.2e}, "
          f"max = {np.max(root_diffs):.2e}")
    print(f"  Score diffs at theta=0: median = {np.median(max_score_diffs):.2e}, "
          f"max = {np.max(max_score_diffs):.2e}")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Use log10 for the x-axis to show the machine-precision scale
    log_diffs = np.log10(root_diffs + 1e-20)  # add tiny offset to avoid log(0)

    ax.hist(log_diffs, bins=30, color=COL_HIST, edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=3)
    ax.axvline(np.log10(np.finfo(float).eps), color=COL_COMP,
               linewidth=1.0, linestyle="--", label="Machine epsilon", zorder=4)

    ax.set_xlabel(
        r"$\log_{10}\,|\hat{\theta}_{\mathrm{comp}} - \hat{\theta}_{\mathrm{IPCW}}|$"
    )
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Root-finding differences across {len(root_diffs)} replications "
        f"($n={n}$)",
        fontsize=11, fontweight="normal"
    )
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig2_mc_roots.{ext}")
    plt.close(fig)
    print(f"  Figure 2 saved.")

    return root_diffs


def main():
    outdir = "."
    n = 100
    theta0 = 0.0
    lam_c = 0.8   # yields ~45% censoring when rate_T = exp(0) = 1

    print("=" * 60)
    print("Score identity verification: PL tail completion = IPCW")
    print("=" * 60)


    print("\nSearching for a representative dataset...")
    rng = np.random.default_rng(42)
    for attempt in range(500):
        Y_ord, Delta_ord = generate_censored_data(n, theta0, lam_c, rng)
        if check_assumptions(Y_ord, Delta_ord):
            cens_rate = 1 - Delta_ord.mean()
            if 0.35 < cens_rate < 0.55:  # aim for ~45%
                break

    n_events = int(Delta_ord.sum())
    n_cens = n - n_events
    print(f"  Dataset found: n={n}, events={n_events}, censored={n_cens} "
          f"({100*n_cens/n:.0f}% censoring)")

    # --- Figure 1 ---
    print("\nGenerating Figure 1 (score curves + difference)...")
    max_diff, root_diff = make_figure1(Y_ord, Delta_ord, "rep", outdir)

    # --- Figure 2 ---
    print(f"\nGenerating Figure 2 (Monte Carlo, {500} replications)...")
    root_diffs = make_figure2(n, theta0, lam_c, 500, outdir)

    print("\nDone. All figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
