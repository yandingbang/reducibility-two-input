# -*- coding: utf-8 -*-
"""Supplementary Figure S1 -- how large a departure q95 can see.

Each run is one synthetic field with a known analytic sigma_true, measured once
under the deployed merged protocol (40 anchors, 15 equispaced angles, 6 repeats,
noise resampled from the residual pool).  A run counts as detected when its
sigma_true_hat exceeds the q95 of that same run; tau_1 plays no part.  The x axis
divides the analytic sigma_true by that run's q95.

Two structurally different field families,
  single ridge + linear term   resp = sc f(a x + b y) + lam sc (-b x + a y)
  two ridges                   resp = sc f(a x + b y) + lam scg g(c x + d y)
each at two noise levels (eps x 1 and x 4 of the calibration), four curves in all.
Colour is the noise level (blue x 1, terracotta x 4); line style and marker are the
family (solid with circles, dashed with squares).  Left out: eps x 0.25, where the
read-out error of the operator exceeds q95 and the curve starts at 54% (brief
section 08-2), and lam = 0, whose sigma_true = 0 has no place on a log axis.

The key sits above the plot in two columns: inside, four rows could not clear the
80% line.  The 80% crossings are printed for the caption, per curve and with each
family's two levels pooled; the range is not drawn (user decision, 2026-09-14).

Data, rerun on 2026-09-16 by data/_resolution_v4.py and data/_resolution_tworidge.py on the
99-angle candidate grid, with the deployed N_FIELDSIM 6400 (the 2026-09-11 runs used 90
angles and 1600); each file is 800 runs, 100 fields x 8 lam.  The pooled 80% crossings barely
moved -- single ridge 1.47 -> 1.46, two ridges 1.39 -> 1.40 -- so the coefficient near 1.5 is
unchanged; the per-level crossings moved by up to 0.07 in either direction.
  _resolution_v4_1.npz, _resolution_v4_4.npz   single ridge + linear term
  _res_tworidge_1.npz,  _res_tworidge_4.npz    two ridges
  rows: eps multiplier, lam, field, sigma_true, sigma_noise, q95, sigma_true_hat, detected
Binning as the draft (data/plot_families.py): 13 log bins on [0.05, 4], then
[4, 8) and [8, inf); a bin is drawn at its median x when it holds at least 8 runs
(one curve), and pooled crossings use at least 12.

Arial 7 pt, black text (the Nature research figure guide lists coloured text among
things to avoid), lines of at least 1 pt, drawn at final size, white background.
Output: figS1_power.png / .svg, 88 mm wide
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data"
#          eps multiplier, legend label,                  colour
LEVELS = ((1.0, r"Noise $\varepsilon\times1$", "#3f6fa0"),
          (4.0, r"Noise $\varepsilon\times4$", "#c0603a"))
#            legend name,                  file pattern,             line style,       marker
FAMILIES = (("Single ridge + linear term", "s1_linear_%g.npz", "-",              "o"),
            ("Two ridges",                 "s1_tworidge_%g.npz",  (0, (3.2, 1.6)), "s"))
EDG = np.concatenate([np.geomspace(0.05, 4.0, 13), [8.0, np.inf]])
LW, MS = 1.2, 2.8


def load(pattern):
    A = np.vstack([np.load(DATA / (pattern % m))["rows"] for m, _, _ in LEVELS])
    return A[:, 3] / np.maximum(A[:, 5], 1e-12), A[:, 7], A[:, 0]


def curve(x, det, minn):
    xs, ys, ns = [], [], []
    for lo, hi in zip(EDG[:-1], EDG[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= minn:
            xs.append(float(np.median(x[m])))
            ys.append(100.0 * float(det[m].mean()))
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


def cross80(xs, ys):
    for i in range(len(xs) - 1):
        if ys[i] < 80 <= ys[i + 1]:
            return xs[i] + (80 - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i])
    return float("nan")


def main():
    W, H = 88 * MM, 69 * MM
    L, B, R, T = 0.46, 0.40, 0.10, 0.36          # T holds the two-row key above the plot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    ax.set_xscale("log")
    ax.set_xlim(0.05, 9.0)
    ax.set_ylim(-3, 104)

    # the two reference levels: 80% detection, and the nominal 5% of q95
    ax.axhline(80, color=C.C_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(5, color=C.C_MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.text(8.6, 81.5, "80%", ha="right", va="bottom", fontsize=FS, color="black")
    ax.text(8.6, 6.5, "5%", ha="right", va="bottom", fontsize=FS, color="black")

    report = []
    for name, pattern, ls, mk in FAMILIES:
        x, det, lev = load(pattern)
        for m, _, col in LEVELS:                           # one family at one level
            k = lev == m
            xs, ys, ns = curve(x[k], det[k], 8)
            ax.plot(xs, ys, color=col, lw=LW, ls=ls, marker=mk, ms=MS, mec="none", zorder=3)
            report.append((name, "eps x%g" % m, cross80(xs, ys)))
            low, high = k & (x >= 0.05) & (x < 0.3), k & (x >= 2)
            print("%-27s eps x%g  runs %3d | 0.05 <= x < 0.3: n %3d, detected %.1f%% | "
                  "x >= 2: n %3d, detected %.1f%%"
                  % (name, m, k.sum(), low.sum(), 100 * det[low].mean(), high.sum(),
                     100 * det[high].mean()))
        xs, ys, _ = curve(x, det, 12)                      # pooled: printed, not drawn
        report.append((name, "pooled", cross80(xs, ys)))

    # key above the plot: colour for the noise level (first column), grey line style
    # and marker for the family (second column)
    handles = [Line2D([], [], color=col, lw=LW) for _, _, col in LEVELS] + \
              [Line2D([], [], color=C.C_MUTED, lw=LW, ls=ls, marker=mk, ms=MS, mec="none")
               for _, _, ls, mk in FAMILIES]
    labels = [lab for _, lab, _ in LEVELS] + [name for name, _, _, _ in FAMILIES]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.5,
              labelspacing=0.35, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)

    ax.set_xticks([0.1, 0.3, 1, 2, 4, 8])
    ax.set_xticklabels(["0.1", "0.3", "1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(direction="in", length=2)
    # sigma_grad since 2026-09-18 (the author): the x axis is the analytic spread of the
    # gradient directions, and the name keeps it apart from the estimate sigma_hat_true on y
    ax.set_xlabel(r"True departure, $\sigma_{\rm grad}\,/\,q_{95}$", labelpad=2)
    ax.set_ylabel(r"Detected, $\hat\sigma_{\rm true}>q_{95}$ (%)", labelpad=2)

    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("figS1_power.%s" % ext), dpi=400, facecolor="white")
    plt.close(fig)
    print("figS1_power  %.0f x %.0f mm" % (W * 25.4, H * 25.4))
    for name, level, c in report:
        print("   80%% crossing  %-27s %-7s %.2f" % (name, level, c))


if __name__ == "__main__":
    main()
