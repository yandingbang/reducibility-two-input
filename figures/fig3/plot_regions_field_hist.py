# -*- coding: utf-8 -*-
"""Figure 3 -- wound healing (Fisher-KPP scratch assay): stage one.

Adapted from the author's v7_biology/plot_regions_field_hist.py (this file began
as a copy of it).  Kept from that script: the three windows on one broken
ln Pi1 axis and one shared ln Pi2 axis, each column as wide as its window's
spread of anchors, the window names with (Pi1 << 1, ~ 1, >> 1) under them, the
colours, and the angle distributions on a 0-90 deg axis below, one column each.

Changed, to match Figure 2 (Figure2/fig2_pipe.py, panel d):
  * the data are one realisation of the current protocol
    (deploy_stop/figures/fig2_method_data.npz), not the DRL read-out of
    test_scratch_regions.csv; stage one is taken from Figure2/fig2_common.py
  * natural logarithms, as Figure 2's ln Re
  * Arial at 7 pt with upright Times New Roman Greek, every line at least 1 pt,
    drawn at final size (95 mm wide)
  * the arrows are the directions read at the 40 anchors, unwrapped to within
    90 deg of the window's mean, 3 mm long on a 1 pt shaft; the axis limits are
    solved in inches so every arrow tip stays inside its column
  * the histograms carry no numbers: stage one's outcome is Figure 1c's
    condition in a box, and only the mean direction is labelled; sigma_alpha,
    sigma_noise, sigma_true and q95 go to a table in the paper
  * one density axis for the three windows and a 2.5 deg KDE bandwidth, as in
    Figure 2 (that script used 3.0 deg and a separate scale per window)
  * the angle axis has 6 deg of room on each side: the readings run from -0.5 to
    90.7 deg, and a strict 0-90 deg axis cut the fast-migration peak in half;
    ticks at 0, 30, 60 and 90 deg, since 45 and 60 collide at 7 pt

Output: fig3_stage1.png / .svg
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent             # resolved: run as a bare file name,
sys.path.insert(0, str(ROOT.parent))   # parent of "." would be "." again
import common as C                                   # noqa: E402  fonts, data, stages

MM = 1 / 25.4
LABEL_FS = 7                     # every label at 7 pt, the ceiling of Nature's figure guide
UNITS = ("u7", "u6", "u5")                                  # slow, intermediate, fast
RCOL = ("#c0603a", "#6b7280", "#3f6fa0")                    # the author's window colours
RNAME = ("Slow migration", "Intermediate", "Fast migration")
RSUB = (r"$(\Pi_1\ll 1)$", r"$(\Pi_1\sim 1)$", r"$(\Pi_1\gg 1)$")
# stage one's outcome as Figure 1c writes the condition; same text as fig2_pipe.py
STAGE1_CONDITION = {
    "REDUCIBLE": r"$\hat\sigma_{\rm true}<\tau_1$" "\n" r"and $q_{95}<\tau_1$",
    "NOT-RED": r"$\hat\sigma_{\rm true}>\max(\tau_1,q_{95})$",
}
KDE_BW = 2.5                                                # degrees, as in Figure 2
ARROW_IN = 0.12                                             # arrow length at final size
SHAFT_IN = 1.0 / 72                                         # 1 pt, NC's thinnest line
HEAD_IN = (0.042, 0.0525, 0.04725)       # head width, length, axis length (inches)
EDGE_IN = 0.06                           # outermost anchor or arrow tip to an axis edge

d = C.load()
S = {u: C.stages(d, u) for u in UNITS}


def unwrap_deg(ang_rad, centre_deg):
    """Axial angles (mod 180) moved to within 90 deg of the window's mean."""
    a = np.degrees(np.asarray(ang_rad))
    return centre_deg + ((a - centre_deg + 90.0) % 180.0) - 90.0


def kde(samples, grid, bw):
    z = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * z ** 2).mean(1) / (bw * math.sqrt(2.0 * math.pi))


def axes_in(fig, W, H, left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def pill(ax, x, y, verdict, text, ha="left", va="top"):
    fc, ec, tc = C.VERDICT[verdict]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=LABEL_FS, color=tc,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=fc, ec=ec, lw=1.0))


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), transparent=True)
    size = fig.get_size_inches() * 25.4
    plt.close(fig)
    print("%-14s %.0f x %.0f mm" % (name, size[0], size[1]))


def extent(scale, pos, lead):
    """(low, high) in inches of anchors at pos (ln units) whose arrows reach lead
    inches along this axis, with EDGE_IN of room on both sides."""
    return (float(np.min(pos * scale + np.minimum(lead, 0.0))) - EDGE_IN,
            float(np.max(pos * scale + np.maximum(lead, 0.0))) + EDGE_IN)


def solve_scale(total_in, groups):
    """Inches per ln unit at which the groups' extents add up to total_in."""
    lo, hi = 1e-3, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        used = sum(b - a for a, b in (extent(mid, p, l) for p, l in groups))
        lo, hi = (mid, hi) if used < total_in else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    W = 95 * MM
    L_M, R_M, BG, GAP = 0.38, 0.05, 0.10, 0.28   # BG: the breaks between field columns
    TITLE_F, FIELD, BELOW1 = 0.26, 1.05, 0.32
    # the histograms run to YTOP ymax so stage one's condition has its own band
    # above the mean labels; HIST grows with YTOP so the curves keep the scale of
    # Figure 2's, 0.80 in per 1.45 ymax
    YTOP = 1.89
    TITLE_H, HIST, BELOW2, XLAB = 0.21, 0.80 * (YTOP + 0.10) / 1.45, 0.16, 0.16
    H = 0.02 + TITLE_F + FIELD + BELOW1 + TITLE_H + HIST + BELOW2 + XLAB
    y_hist = XLAB + BELOW2
    y_field = y_hist + HIST + TITLE_H + BELOW1

    prep = []
    for u in UNITS:
        s = S[u]
        centre = math.degrees(s["abar"])
        ang = unwrap_deg(s["ang"], centre)
        rad = np.radians(ang)
        prep.append(dict(s=s, centre=centre, ang=ang,
                         dx=ARROW_IN * np.cos(rad), dy=ARROW_IN * np.sin(rad)))

    fig = plt.figure(figsize=(W, H))

    # ---- the directions read at the anchors, on one broken ln Pi1 axis --------------
    sx = solve_scale(W - L_M - R_M - 2 * BG, [(p["s"]["CX"], p["dx"]) for p in prep])
    all_y = np.concatenate([p["s"]["CY"] for p in prep])
    all_dy = np.concatenate([p["dy"] for p in prep])
    sy = solve_scale(FIELD, [(all_y, all_dy)])
    y_lo, y_hi = extent(sy, all_y, all_dy)
    brk = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=5, linestyle="none",
               color="black", mew=1.0, clip_on=False, zorder=6)
    left = L_M
    for k, (p, name, sub, col) in enumerate(zip(prep, RNAME, RSUB, RCOL)):
        s = p["s"]
        a, b = extent(sx, s["CX"], p["dx"])
        ax = axes_in(fig, W, H, left, y_field, b - a, FIELD)
        rad = np.radians(p["ang"])
        ax.scatter(s["CX"], s["CY"], s=2, color=col, alpha=0.55, edgecolor="none",
                   zorder=4)
        ax.quiver(s["CX"], s["CY"], np.cos(rad), np.sin(rad), color=col, alpha=0.9,
                  angles="uv", pivot="tail", units="inches", scale_units="inches",
                  scale=1.0 / ARROW_IN, width=SHAFT_IN,
                  headwidth=HEAD_IN[0] / SHAFT_IN, headlength=HEAD_IN[1] / SHAFT_IN,
                  headaxislength=HEAD_IN[2] / SHAFT_IN, zorder=3)
        ax.set_xlim(a / sx, b / sx)
        ax.set_ylim(y_lo / sy, y_hi / sy)
        ax.xaxis.set_major_locator(MaxNLocator(max(1, int((b - a) / 0.32)),
                                               steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(direction="in", length=2)
        if k == 0:
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_ylabel(r"$\ln(n_0/K)$", labelpad=1)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
        # the breaks, drawn where the bottom spines stop
        if k < 2:
            ax.plot([1], [0], transform=ax.transAxes, **brk)
        if k > 0:
            ax.plot([0], [0], transform=ax.transAxes, **brk)
        cx = (left + (b - a) / 2) / W
        fig.text(cx, (y_field + FIELD + 0.17) / H, name, ha="center", va="baseline",
                 fontsize=7, fontweight="bold", color=col)
        fig.text(cx, (y_field + FIELD + 0.05) / H, sub, ha="center", va="baseline",
                 fontsize=LABEL_FS, color=col)
        left += (b - a) + BG
    fig.text((L_M + W - R_M) / 2 / W, (y_field - 0.14) / H, r"$\ln[D_n/(rL^2)]$",
             ha="center", va="top", fontsize=7, color=C.C_TEXT)

    # ---- the readings' distributions, one angle axis and one density axis ----------
    xlo = min(0.0, min(p["ang"].min() for p in prep)) - 6.0
    xhi = max(90.0, max(p["ang"].max() for p in prep)) + 6.0
    grid = np.linspace(xlo, xhi, 1600)
    for p in prep:
        p["dens"] = kde(p["ang"], grid, KDE_BW)
    ymax = max(p["dens"].max() for p in prep)
    colw = (W - L_M - R_M - 2 * GAP) / 3
    for k, (p, name, col) in enumerate(zip(prep, RNAME, RCOL)):
        s, left = p["s"], L_M + k * (colw + GAP)
        axh = axes_in(fig, W, H, left, y_hist, colw, HIST)
        axh.fill_between(grid, p["dens"], color=col, alpha=0.80, linewidth=0, zorder=3)
        axh.plot(grid, p["dens"], color=col, lw=1.0, zorder=4)
        axh.vlines(p["ang"], -0.085 * ymax, -0.02 * ymax, color=col, lw=1.0,
                   alpha=0.7, zorder=4)
        # the dashed mean stops under its label
        axh.vlines(p["centre"], 0, 1.08 * ymax, linestyles="--", lw=1.0,
                   color=col, alpha=0.55, zorder=2)
        right = p["centre"] < 45.0
        axh.text(p["centre"] + (2.5 if right else -2.5), 1.10 * ymax,
                 r"$\bar\alpha \approx %.0f$°" % (p["centre"] % 180.0),
                 ha="left" if right else "right", va="bottom", fontsize=LABEL_FS,
                 color=col, style="italic", fontweight="bold", zorder=5)
        pill(axh, 0.06, 0.97, s["c1"], STAGE1_CONDITION[s["c1"]])
        axh.set_xlim(xlo, xhi)
        axh.set_ylim(-0.10 * ymax, YTOP * ymax)
        axh.set_xticks([0, 30, 60, 90])
        axh.set_xticklabels(["0°", "30°", "60°", "90°"])
        axh.set_yticks([0])
        axh.set_yticklabels(["0"])
        axh.tick_params(direction="in", length=2)
        # a title over each histogram, since these columns do not line up with the
        # fields above; at pad 3 pt the "g" of "migration" came within 0.5 mm of
        # the condition box, which starts at the axes top
        axh.set_title(name, fontsize=7, fontweight="bold", color=col, pad=5)
        if k == 0:
            axh.set_ylabel("Density", labelpad=1)
    fig.text((L_M + 1.5 * colw + GAP) / W, 0.02 / H,
             r"Recovered direction angle $\alpha$", ha="center", va="bottom",
             fontsize=7, color=C.C_TEXT)
    save(fig, "fig3_stage1")
    C.compare_published(S)


if __name__ == "__main__":
    main()
