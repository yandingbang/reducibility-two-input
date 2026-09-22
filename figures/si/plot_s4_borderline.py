# -*- coding: utf-8 -*-
"""Supplementary Figure S4 -- a picture cannot grade a collapse.

Data: data/s4_borderline.npz, written by data/s4_borderline.py.  One synthetic field (exp
ridge, seed 51300003), the same 40 anchors and the same noise draw in every panel; only the
transverse departure lam grows: 0, 0.010, 0.020, 0.035, 0.060, 0.120.  Each panel is the
deployed stage-two collapse plot: the averaged centre responses against the chosen group
coordinate ln u_hat, with the polynomial the degree ladder selected.  The number in the
corner is the collapse residual rho, and under it the stage-two verdict at tau = 5% with the
adopted floor tau_min = eps_pool / (sqrt(n) s_y) (0.0087 in every panel).

The point of the figure is that the eye has nothing to go on: rho grows sixfold across the
first five panels, which look alike.  So nothing in the drawing hints at the verdict: every
panel uses the same neutral points and curve, the same axes, and black text.

Six panels in two rows, 180 mm wide: in a single row each panel would be 26 mm wide, too
small for a fair look at the scatter.  Arial 7 pt, lines of at least 1 pt, drawn at final
size, white background.
Output: figS4_borderline.png / .svg
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data" / "s4_borderline.npz"
INK = "#1f2937"
POINT_S = 6.0                                               # marker area, pt^2
VERDICT = {"reduce": "Reducible", "not-red": "Not reducible", "undet": "Undetermined"}

W = 180 * MM
L, R, CG = 0.50, 0.08, 0.14                                 # inches: left, right, column gap
B, AXH, RG, TITLE, T = 0.40, 1.30, 0.10, 0.17, 0.04         # bottom, axes, row gap, title, top
NCOL = 3


def main():
    d = np.load(DATA, allow_pickle=True)
    lams = d["lams"]
    n = len(lams)
    us = [d["u%d" % k] for k in range(n)]
    ys = [d["y%d" % k] for k in range(n)]
    xlo, xhi = min(u.min() for u in us), max(u.max() for u in us)
    ylo, yhi = min(y.min() for y in ys), max(y.max() for y in ys)
    xpad, ypad = 0.05 * (xhi - xlo), 0.06 * (yhi - ylo)

    nrow = (n + NCOL - 1) // NCOL
    colw = (W - L - R - (NCOL - 1) * CG) / NCOL
    H = B + nrow * AXH + (nrow - 1) * RG + nrow * TITLE + T
    fig = plt.figure(figsize=(W, H))
    print("figS4_borderline  %.0f x %.0f mm" % (W * 25.4, H * 25.4))
    print("%7s %9s %9s %9s %5s  %s" % ("lam", "rho", "tau_min", "rho_true", "deg", "verdict"))
    for k in range(n):
        row, col = divmod(k, NCOL)
        left = L + col * (colw + CG)
        bottom = B + (nrow - 1 - row) * (AXH + RG + TITLE)
        ax = fig.add_axes([left / W, bottom / H, colw / W, AXH / H])
        u, y, p = us[k], ys[k], d["p%d" % k]
        m, sd = u.mean(), u.std()
        ug = np.linspace(u.min(), u.max(), 300)
        ax.plot(ug, np.polyval(p, (ug - m) / sd), color=C.C_MUTED, lw=1.0, zorder=2)
        ax.scatter(u, y, s=POINT_S, color=INK, edgecolor="none", zorder=3)
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        # steps of 1, 2 or 5 only: the default gave 0, 1.5, 3.0, 4.5
        ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.tick_params(direction="in", length=2)
        if col:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"Response $y$", labelpad=2)
        if row < nrow - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$\ln\hat u$", labelpad=1)
        ax.set_title(r"$\lambda=%.3f$" % lams[k], fontsize=FS, pad=3)
        rho, verdict = float(d["rho%d" % k]), str(d["verdict%d" % k])
        ax.text(0.96, 0.94, r"$\rho=%.4f$" % rho + "\n" + VERDICT[verdict], transform=ax.transAxes,
                ha="right", va="top", fontsize=FS, color="black", linespacing=1.4)
        print("%7.3f %9.4f %9.4f %9.4f %5d  %s" % (lams[k], rho, float(d["tau_min%d" % k]),
                                                   float(d["rho_true%d" % k]), int(d["deg%d" % k]),
                                                   VERDICT[verdict]))

    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("figS4_borderline.%s" % ext), dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
