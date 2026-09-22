# -*- coding: utf-8 -*-
"""Supplementary Figure S6 -- the read-out error of the benchmarks has two terms, both first order in probe size.

Data: data/s6_probe_scale.npz, written by data/s6_probe_scale.py.  Noise-free.  The 40 anchors
of each of the eight benchmarks stay where the deployed pipeline puts them; the probe half-axes
are scaled to 1, 1/2, 1/4, 1/8, 1/16 and 1/40 of the deployed probe (which reaches the edge of
the box).  Each probe profile is read at the deployed 15 equispaced angles and at all 99
candidate angles.  The truth is the analytic gradient direction at each anchor.

a  figS6a_convergence: |sigma_alpha,clean - sigma_grad|, the deployed read-out (15 angles) against the
   probe size, log x and symmetric-log y.  Seven benchmarks fall with the probe; the dashed grey
   segment has slope 1.  Low Re is the flat line on 0: its contours run along y, so the crossing
   sits at pi/2, which since SIZE = 99 is one of the 15 queried angles (index 49 = 7 x 7), and the
   read-out returns the truth exactly at every probe size.  y is linear below 2.5e-5, under the
   smallest non-zero error, so that the zero has a place; the data span 7e-5 (high Re at 1/40,
   where the read-out crosses the truth) to 2.06 deg (medium Re at the deployed probe).
b  figS6b_terms: at the deployed probe, the per-anchor RMS read-out error split in two:
     chord          99-angle read-out minus the analytic direction: the chord to the point of
                    equal response leaves the tangent where the contour bends;
     interpolation  15-angle read-out minus 99-angle read-out: the zero crossing interpolated
                    between 15 angles.
   Low Re, microfluidic and buckling depend on one linear combination of the two log inputs
   (64 e^-x, e^(-x + y/3), pi^2 / (pi^2 + e^(2x + y))), so their contours are straight and the
   chord term is exactly 0; what their 99-angle read-out still shows (0 for low Re, 0.0102 and
   0.0062 deg RMS) is interpolation at 1.8-degree spacing, and it bounds the interpolation left
   inside the chord bars of the other five.  Low Re's interpolation term is 0 as well, for the
   reason panel a gives, and both of its terms are written out instead of drawn.  Horizontal
   bars on a log axis start at the left edge of the axis.

Marker shape is the system as in Figure 5 (circle rough pipe, square wound healing, triangle
microfluidic, diamond column buckling); within a system the regimes are solid, dashed, dotted.
Benchmark names are short (the wound regimes named by their migration speed) so that the key of
a and the category labels of b fit the 88 mm width.
Arial 7 pt, black text, lines of at least 1 pt, drawn at final size, white background; the
axes height of the figS1_power frame, with a taller top margin for the four-row key of a, and b
the same size so the two files line up.
Outputs: figS6a_convergence.png / .svg and figS6b_terms.png / .svg, 88 mm wide each
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data" / "s6_probe_scale.npz"
INK = "#1f2937"
GREY = "#9aa3ad"
DOTTED, DASHED = (0, (1, 1.5)), (0, (3.2, 1.6))
#            npz name,                 label,                     marker, line style
BENCH = (("Rough-pipe, low Re",       r"Rough pipe, low $Re$",    "o", "-"),
         ("Rough-pipe, medium Re",    r"Rough pipe, medium $Re$", "o", DASHED),
         ("Rough-pipe, high Re",      r"Rough pipe, high $Re$",   "o", DOTTED),
         ("Microfluidic",             "Microfluidic",             "^", "-"),
         ("Wound, slow migration",    "Wound, slow",              "s", "-"),
         ("Wound, intermediate",      "Wound, intermediate",      "s", DASHED),
         ("Wound, fast migration",    "Wound, fast",              "s", DOTTED),
         ("Column buckling",          "Column buckling",          "D", "-"))
STRAIGHT = {"Rough-pipe, low Re", "Microfluidic", "Column buckling"}   # one linear combination
FRACTION = {1.0: "1", 0.5: "1/2", 0.25: "1/4", 0.125: "1/8", 0.0625: "1/16", 0.025: "1/40"}
LW, MS = 1.2, 3.0
W = 88 * MM
B, R = 0.40, 0.10
AXH = 69 * MM - 0.40 - 0.36                                 # the axes height of figS1_power
T = 0.64                                                    # room for the four-row key of a
H = B + AXH + T


def key(ax, handles, labels, ncol):
    return ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=ncol,
                     frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.5,
                     labelspacing=0.35, columnspacing=1.2, borderaxespad=0.0, borderpad=0.0)


def save(fig, leg, name):
    fig.canvas.draw()
    fb = fig.bbox
    boxes = [leg.get_window_extent()] + [t.get_window_extent() for ax in fig.axes
                                          for t in ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    print("%s  %.0f x %.1f mm; text clears the left edge by %.1f mm, the right by %.1f mm, the top by %.1f mm"
          % (name, W * 25.4, H * 25.4, (x0 - fb.x0) / fig.dpi * 25.4, (fb.x1 - x1) / fig.dpi * 25.4,
             (fb.y1 - y1) / fig.dpi * 25.4))
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), dpi=400, facecolor="white")
    plt.close(fig)


def dwrap(a):
    return (np.degrees(a) + 90.0) % 180.0 - 90.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def panel_a(d, idx):
    L = 0.50
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, AXH / H])
    lams = d["lams"]
    err = {n: np.abs(d["sig15"][idx[n]] - d["truth"][idx[n], 0]) for n, _, _, _ in BENCH}
    ylo, yhi = 2.5e-5, 4.0                                  # ylo: the linear band, under every datum
    nz = np.concatenate([v[v > 0] for v in err.values()])
    assert nz.min() > ylo and max(v.max() for v in err.values()) < yhi
    ax.set_xscale("log")
    # symlog since 2026-09-16: low Re reads its truth exactly at every probe size, and 0 has no
    # place on a log axis.  The linear band sits below the smallest non-zero error
    ax.set_yscale("symlog", linthresh=ylo, linscale=0.3)
    ax.set_xlim(0.019, 1.3)
    ax.set_ylim(0.0, yhi)
    handles, labels = [], []
    print("panel a: |sigma_alpha - sigma_true| at 15 angles, deg, at probe sizes %s"
          % ", ".join(FRACTION[float(l)] for l in lams))
    for n, lab, mk, ls in BENCH:
        e = err[n]
        ax.plot(lams, e, color=INK, lw=LW, ls=ls, marker=mk, ms=MS, mec="none", zorder=3)
        handles.append(Line2D([], [], color=INK, lw=LW, ls=ls, marker=mk, ms=MS, mec="none"))
        labels.append(lab)
        print("   %-24s %s | ratios between neighbours %s"
              % (n, "  ".join("%.5f" % v for v in e),
                 " ".join("%.2f" % (e[i] / e[i + 1]) if e[i + 1] > 0 else "   -"
                          for i in range(len(e) - 1))))
    # slope-1 guide in the empty lower right, a factor 5 under the lowest curve at probe size 1
    y1 = min(err[n][0] for n in err if err[n][0] > 0) / 5.0
    xs = np.array([0.35, 1.0])
    ax.plot(xs, y1 * xs, color=C.C_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(0.33, y1 * 0.35, "Slope 1", ha="right", va="center", fontsize=FS, color="black")
    leg = key(ax, handles, labels, ncol=2)
    ax.set_xticks([float(l) for l in lams])
    ax.set_xticklabels([FRACTION[float(l)] for l in lams])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["0"] + [r"$10^{%d}$" % k for k in (-4, -3, -2, -1, 0)])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(direction="in", length=2)
    ax.set_xlabel("Probe size, fraction of the deployed probe", labelpad=2)
    # sigma_alpha,clean against sigma_grad since 2026-09-18 (the author): everything here is
    # noise-free, and the truth is the analytic spread of the gradient directions
    ax.set_ylabel(r"Read-out error, $|\sigma_{\alpha,\,\rm clean}-\sigma_{\rm grad}|$ (°)", labelpad=2)
    save(fig, leg, "figS6a_convergence")


def panel_b(d, idx):
    L = 1.24                                                # the longest category name needs 1.1 in
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, AXH / H])
    xmin, xmax = 3e-4, 20.0
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    hbar, gap = 0.34, 0.04                                  # bar height and the gap between the pair
    nall = int(d["n_all"]) if "n_all" in d.files else 90     # candidate angles: 99 since 2026-09-16
    ys = np.arange(len(BENCH))
    ax.set_ylim(len(BENCH) - 0.45, -0.55)
    print("panel b: per-anchor RMS read-out error at the deployed probe, deg")
    for y, (n, lab, _, _) in zip(ys, BENCH):
        u = idx[n]
        chord90 = rms(dwrap(d["ang90"][u, 0] - d["a_true"][u]))
        interp = rms(dwrap(d["ang15"][u, 0] - d["ang90"][u, 0]))
        yc, yi = y - (hbar + gap) / 2, y + (hbar + gap) / 2
        if n in STRAIGHT:
            ax.text(xmin * 1.35, yc, "0 (straight contours)", ha="left", va="center", fontsize=FS,
                    color="black")
            print("   %-24s chord 0 (straight contours; %d-angle read-out %.4f)  interpolation %.4f"
                  % (n, nall, chord90, interp))
        else:
            ax.barh(yc, chord90 - xmin, left=xmin, height=hbar, color=INK, lw=0, zorder=3)
            print("   %-24s chord %.4f  interpolation %.4f" % (n, chord90, interp))
        if interp > 0:
            ax.barh(yi, interp - xmin, left=xmin, height=hbar, color=GREY, lw=0, zorder=3)
        else:
            # low Re since 2026-09-16: its crossing is at pi/2, one of the 15 queried angles
            ax.text(xmin * 1.35, yi, "0 (crossing on a queried angle)", ha="left", va="center",
                    fontsize=FS, color="black")
    leg = key(ax, [Patch(color=INK, lw=0), Patch(color=GREY, lw=0)],
              ["Chord (bent contours)", "Interpolation (15 angles)"], ncol=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _, _ in BENCH])
    ax.tick_params(axis="y", length=0)                      # categories: no tick marks
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", direction="in", length=2)
    ax.set_xlabel("Per-anchor read-out error, RMS (°)", labelpad=2)
    save(fig, leg, "figS6b_terms")


def main():
    d = np.load(DATA, allow_pickle=True)
    idx = {str(n): i for i, n in enumerate(d["names"])}
    assert set(idx) == {b[0] for b in BENCH}
    assert STRAIGHT <= set(idx) and all(abs(d["truth"][idx[n], 0]) < 1e-9 for n in STRAIGHT)
    panel_a(d, idx)
    panel_b(d, idx)


if __name__ == "__main__":
    main()
