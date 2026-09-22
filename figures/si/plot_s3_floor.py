# -*- coding: utf-8 -*-
"""Supplementary Figure S3 -- does tau_min describe the noise floor of the collapse residual?

Data: data/s3_floor_800_200.npz, written by data/s3_floor_calibration.py: the deployed
stage-two path (merged protocol, n = 6, window search around alpha_bar, rho_poly with the
degree ladder) with the adopted floor tau_min = eps_pool / (sqrt(n) s_y).  Two panels, two
files.

a  figS3a_floor: exactly reducible fields (the S2 fields; at eps x 1 and x 4 the very same
   campaigns), 800 per noise level at eps x 0.25, x 1 and x 4.  Each point is one run:
   the observed residual rho against its floor tau_min, log-log, with the line
   rho = tau_min.  If the floor is right the clouds sit on the line whatever the noise.

b  figS3b_estimate: fields with a genuine departure, resp = sc f(a x + b y) + lam sc
   (-b x + a y), lam 0.02 to 0.40, 200 fields per lam, at eps x 1 and x 4.  Each point is
   one run: rho_true = sqrt(rho^2 - tau_min^2) against rho_clean, the best rho_poly over a
   181-direction scan of the noiseless centre responses.  The dotted lines are the
   tolerance tau = 5% on both axes, so a point below the horizontal one and right of the
   vertical one is a field above 5% called reducible, and one above and left of them is a
   field within 5% called not reducible.  The verdict compares rho_true with
   max(tau, tau_min), so the few runs whose tau_min exceeds 5% are undetermined instead;
   their count is printed.
   Both axes are symmetric-log, linear inside 0.01, as in Figure 5: rho_true is clipped
   to 0 in part of the runs, and on linear axes the handful of runs reaching 0.35 squeezed
   the 5% boundary, where the verdict errors are, into a corner.  Both axes start just
   below 0 so that the two zero ticks do not meet at the corner.

Colours: eps x 1 blue and x 4 terracotta as in S1 and S2, eps x 0.25 ink.  Arial 7 pt,
black text, lines of at least 1 pt, drawn at final size, white background, the frame of
S1 and S2.
Outputs: figS3a_floor.png / .svg and figS3b_estimate.png / .svg, 88 mm wide each
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data" / "s3_floor_800_200.npz"
#          eps multiplier: key label,                         colour
LEVELS = {0.25: (r"Noise $\varepsilon\times0.25$", "#1f2937"),
          1.0: (r"Noise $\varepsilon\times1$", "#3f6fa0"),
          4.0: (r"Noise $\varepsilon\times4$", "#c0603a")}
DOT_S, DOT_ALPHA, KEY_MS = 3.0, 0.45, 3.2
REF_LS, TAU_LS = (0, (4, 3)), (0, (1, 2))
W, H = 88 * MM, 69 * MM
L, B, R, T = 0.46, 0.40, 0.10, 0.36                         # the frame of S1 and S2


def frame():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    return fig, ax


def key(ax, handles, labels):
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.5,
              labelspacing=0.35, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)


def dot(col):
    return Line2D([], [], ls="none", marker="o", ms=KEY_MS, mfc=col, mec="none")


def fixed_ticks(ax, ticks):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FixedFormatter(["%g" % v for v in ticks]))
        axis.set_minor_locator(NullLocator())


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), dpi=400, facecolor="white")
    plt.close(fig)
    print("%s  %.0f x %.0f mm" % (name, W * 25.4, H * 25.4))


def panel_a(A, c):
    fig, ax = frame()
    null = A[A[:, c["kind"]] == 0]
    x, y = null[:, c["tau_min"]], null[:, c["rho"]]
    # the same range on both axes, out to the nearest quarter decade
    lo = 10 ** (np.floor(np.log10(min(x.min(), y.min())) * 4) / 4)
    hi = 10 ** (np.ceil(np.log10(max(x.max(), y.max())) * 4) / 4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=C.C_MUTED, lw=1.0, ls=REF_LS, zorder=1)

    print("panel a: exactly reducible fields, rho / tau_min; axes %.2g-%.2g" % (lo, hi))
    for m in (4.0, 1.0, 0.25):                              # smallest floors drawn last
        s = null[:, c["eps_mult"]] == m
        ax.scatter(x[s], y[s], s=DOT_S, c=LEVELS[m][1], alpha=DOT_ALPHA, linewidths=0, zorder=3)
        r = y[s] / x[s]
        print("   eps x%-5g n %d  median %.3f  IQR %.3f-%.3f  95th %.3f  tau_min %.4f-%.4f"
              % (m, s.sum(), np.median(r), np.percentile(r, 25), np.percentile(r, 75),
                 np.percentile(r, 95), x[s].min(), x[s].max()))

    order = (0.25, 1.0, 4.0)
    key(ax, [dot(LEVELS[m][1]) for m in order] +
        [Line2D([], [], color=C.C_MUTED, lw=1.0, ls=REF_LS)],
        [LEVELS[m][0] for m in order] + [r"$\rho=\tau_{\min}$"])
    fixed_ticks(ax, [10.0 ** k for k in range(int(np.ceil(np.log10(lo))),
                                             int(np.floor(np.log10(hi))) + 1)])
    ax.tick_params(direction="in", length=2)
    ax.set_xlabel(r"Noise floor $\tau_{\min}$", labelpad=2)
    ax.set_ylabel(r"Collapse residual $\rho$", labelpad=2)
    save(fig, "figS3a_floor")


def panel_b(A, c, tau):
    fig, ax = frame()
    tilt = A[A[:, c["kind"]] == 1]
    x, y = tilt[:, c["rho_clean"]], tilt[:, c["rho_true"]]
    top = max(x.max(), y.max())
    hi = 0.4 if top <= 0.4 else float(np.ceil(top * 10) / 10)
    lo = -0.002
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.3)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([0, hi], [0, hi], color=C.C_MUTED, lw=1.0, ls=REF_LS, zorder=1)
    ax.axvline(tau, color=C.C_MUTED, lw=1.0, ls=TAU_LS, zorder=1)
    ax.axhline(tau, color=C.C_MUTED, lw=1.0, ls=TAU_LS, zorder=1)

    print("panel b: fields with a departure; axes %g-%g, symlog" % (lo, hi))
    for m in (4.0, 1.0):
        s = tilt[:, c["eps_mult"]] == m
        ax.scatter(x[s], y[s], s=DOT_S, c=LEVELS[m][1], alpha=DOT_ALPHA, linewidths=0, zorder=3)
        v = tilt[s, c["verdict"]]
        over = x[s] > tau
        print("   eps x%g  n %d | above 5%% called reduce %d of %d | within 5%% called not-red %d of %d"
              " | undetermined %d | tau_min above 5%% in %d runs | rho_true = 0 in %d runs"
              % (m, s.sum(), np.sum(over & (v == 0)), over.sum(), np.sum(~over & (v == 1)),
                 (~over).sum(), np.sum(v == 2), np.sum(tilt[s, c["tau_min"]] > tau),
                 np.sum(y[s] == 0)))

    order = (1.0, 4.0)
    key(ax, [dot(LEVELS[m][1]) for m in order] +
        [Line2D([], [], color=C.C_MUTED, lw=1.0, ls=REF_LS),
         Line2D([], [], color=C.C_MUTED, lw=1.0, ls=TAU_LS)],
        [LEVELS[m][0] for m in order] + [r"$\rho_{\rm true}=\rho_{\rm clean}$", r"$\tau=5\%$"])
    fixed_ticks(ax, [v for v in (0, 0.01, 0.05, 0.1, 0.2, 0.4) if v <= hi])
    ax.tick_params(direction="in", length=2)
    ax.set_xlabel(r"Noise-free residual $\rho_{\rm clean}$", labelpad=2)
    ax.set_ylabel(r"Noise-corrected residual $\rho_{\rm true}$", labelpad=2)
    save(fig, "figS3b_estimate")


def main():
    d = np.load(DATA, allow_pickle=True)
    c = {str(k): i for i, k in enumerate(d["cols"])}
    panel_a(d["rows"], c)
    panel_b(d["rows"], c, float(d["tau"]))


if __name__ == "__main__":
    main()
