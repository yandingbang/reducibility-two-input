# -*- coding: utf-8 -*-
"""Figure 3 -- wound healing (Fisher-KPP scratch assay): stage two and the verdict.

The panel of Figure 2e (Figure2/fig2_pipe.py, panel_e) for the three wound
windows: T_c r at the 40 anchor centres against the single group
u-hat = Pi1^a Pi2^b at the angle stage two searched, stage two's fitted g, which
of Figure 1c's stage-two conditions holds (light box), the verdict of both
stages under it (dark box), and the points coloured by where they sit in their
window, with one key for all columns.

As in Figure 2e: T_c r itself is plotted, not log(T_c r) as in the author's
v7_biology/plot_c_collapse.py, because stage two fits T_c r; u-hat is written
from this realisation's (a, b); "no common reduction" is not printed, since it
is a verdict given before the verdict.  Pi1 and Pi2 are the notation of the
stage-one panel's subtitles.

Where it differs from Figure 2e, the data decide:
  * T_c r falls with u-hat in every window, so the labels hang from the top
    right, where the data are low; the corner follows the sign of the trend
  * the key reads Pi1 and Pi2, and the window colours are the author's wound
    colours, as in plot_regions_field_hist.py

Output: fig3_stage2.png / .svg
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, data, stages

MM = 1 / 25.4
LABEL_FS = 7                     # every label at 7 pt, the ceiling of Nature's figure guide
UNITS = ("u7", "u6", "u5")                                  # slow, intermediate, fast
RCOL = ("#c0603a", "#6b7280", "#3f6fa0")                    # the author's window colours
RNAME = ("Slow migration", "Intermediate", "Fast migration")
# the four corner colours of the author's plot_three_regimes_collapse_row.py, as
# in Figure 2e: (low, low), (high Pi1, low), (low, high Pi2), (high, high)
POS_CORNERS = ("#eef1f5", "#2e8fdf", "#f15a29", "#8f53a8")
POINT_S = 8                                                 # marker area, pt^2
# stage two's part of Figure 1c's conditions; same text as fig2_pipe.py.  Stage two
# compares rho_hat_true = sqrt(max(rho^2 - tau_min^2, 0)), not rho (v2 in
# framework_pipeline.py; C.stages computes c2 from it), hatted like sigma_hat_true;
# both boxes said rho until 2026-09-15, as Figures 1c and 2e did
STAGE2_CONDITION = {
    "REDUCIBLE": r"$\hat\rho_{\rm true}<\tau$" "\n" r"and $\tau_{\min}<\tau$",
    "NOT-RED": r"$\hat\rho_{\rm true}>\max(\tau,\tau_{\min})$",
}

d = C.load()
S = {u: C.stages(d, u) for u in UNITS}


def position_colours(v1, v2):
    """Bilinear between the four corner colours, as in Figure 2e."""
    a = [np.array(to_rgb(c)) for c in POS_CORNERS]
    p, q = np.asarray(v1)[:, None], np.asarray(v2)[:, None]
    return np.clip((1 - p) * (1 - q) * a[0] + p * (1 - q) * a[1]
                   + (1 - p) * q * a[2] + p * q * a[3], 0, 1)


def group_label(a, b):
    """u-hat = Pi1^a Pi2^b for the subtitle, exponents to two decimals."""
    def power(base, p):
        if abs(p) < 0.005:
            return ""
        if abs(p - 1.0) < 0.005:
            return base
        e = "%d" % round(p) if abs(p - round(p)) < 0.005 else "%.2f" % p
        return r"%s^{%s}" % (base, e)
    factors = [f for f in (power(r"\Pi_1", a), power(r"\Pi_2", b)) if f]
    return r"$\hat{u}\approx %s$" % r"\,".join(factors)


def axes_in(fig, W, H, left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def pill(ax, x, y, verdict, text, ha, va="top"):
    """Stage two's condition, in the verdict's light colours; lines stay left-aligned
    inside the box whichever corner it hangs from."""
    fc, ec, tc = C.VERDICT[verdict]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   multialignment="left", fontsize=LABEL_FS, color=tc,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=fc, ec=ec, lw=1.0))


def solid_pill(ax, x, y, verdict, ha, va="top"):
    """The verdict of both stages: dark fill and white text, unlike a condition."""
    dark = C.VERDICT[verdict][2]
    return ax.text(x, y, verdict, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=LABEL_FS, color="white",
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=dark, ec=dark, lw=1.0))


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), transparent=True)
    size = fig.get_size_inches() * 25.4
    plt.close(fig)
    print("%-14s %.0f x %.0f mm" % (name, size[0], size[1]))


def main():
    W0 = 95 * MM                                     # the stage-one panel's width
    L_M, R_M, GAP = 0.38, 0.05, 0.28
    colw = (W0 - L_M - R_M - 2 * GAP) / 3
    KEY_GAP, KEY, KEY_R = 0.24, 0.30, 0.18           # the one key, right of the columns
    W = W0 - R_M + KEY_GAP + KEY + KEY_R
    # TITLE 0.05 in taller than Figure 2e's: Pi carries a superscript over its
    # subscript here, so the subtitle is taller, and at Figure 2e's spacing its
    # subscripts came within 0.8 mm of the condition box and its exponents within
    # 0.6 mm of the title
    BELOW, AXH, TITLE = 0.32, 1.15, 0.38
    H = BELOW + AXH + TITLE
    # condition box to verdict box, 1 mm as seen: the patch extents leave out the
    # 1 pt edges, half of it on each box
    BOX_GAP = (1.0 * MM + 1.0 / 72) / AXH
    PAD = 0.22 * LABEL_FS / 72 / AXH                 # a box's pad, in axes units
    CLEAR = 1.0 * MM                                 # labels to the nearest data, inches
    INSET = 0.06                                     # labels from the side they hang on
    # drawn at the saving dpi, so the label sizes measured below are the saved ones
    fig = plt.figure(figsize=(W, H), dpi=400)
    placed = []
    for j, (u, name, col) in enumerate(zip(UNITS, RNAME, RCOL)):
        s, left = S[u], L_M + j * (colw + GAP)
        ax = axes_in(fig, W, H, left, BELOW, colw, AXH)
        uu, yy = s["u"], s["yn"]
        # stage two's fitted g under the anchors, coloured by where they sit in
        # the window (no rim: a thin rim is a line under NC's 1 pt)
        m, sd = uu.mean(), uu.std()
        coef = np.polyfit((uu - m) / sd, yy, s["deg"])
        ug = np.linspace(uu.min(), uu.max(), 300)
        gg = np.polyval(coef, (ug - m) / sd)
        ax.plot(ug, gg, color=C.C_MUTED, lw=1.0, zorder=2)
        x0, x1, y0, y1 = s["box"]
        ax.scatter(uu, yy, s=POINT_S, edgecolor="none", zorder=3,
                   c=position_colours((s["CX"] - x0) / (x1 - x0),
                                      (s["CY"] - y0) / (y1 - y0)))
        xspan, yspan = np.ptp(uu), np.ptp(yy)
        ax.set_xlim(uu.min() - 0.06 * xspan, uu.max() + 0.06 * xspan)
        ax.set_ylim(yy.min() - 0.08 * yspan, yy.max() + 0.10 * yspan)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.tick_params(direction="in", length=2)
        ax.set_xlabel(r"$\ln\hat{u}$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$T_c\,r$", labelpad=1)
        # title as in the stage-one panel, the single group under it
        cx = (left + colw / 2) / W
        fig.text(cx, (BELOW + AXH + 0.28) / H, name, ha="center", va="baseline",
                 fontsize=7, fontweight="bold", color=col)
        fig.text(cx, (BELOW + AXH + 0.085) / H,
                 group_label(math.cos(s["a_star"]), math.sin(s["a_star"])),
                 ha="center", va="baseline", fontsize=LABEL_FS, color=C.C_MUTED)
        # the labels hang from the top corner away from the data
        rising = np.corrcoef(uu, yy)[0, 1] > 0
        bx, ha = (INSET, "left") if rising else (1.0 - INSET, "right")
        c2 = C.STAGE2_NAME[s["c2"]]
        placed.append((ax, s, bx, ha, np.r_[uu, ug], np.r_[yy, gg],
                       pill(ax, bx, 0.97, c2, STAGE2_CONDITION[c2], ha)))
    # the verdict of both stages, right under stage two's condition
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax, s, bx, ha, _, _, cond in placed:
        y_low = ax.transAxes.inverted().transform(
            (0.0, cond.get_bbox_patch().get_window_extent(rend).y0))[1]
        boxes.append((cond, solid_pill(ax, bx, y_low - BOX_GAP - PAD,
                                       C.JOINT_NAME[s["jt"]], ha)))
    # raise the top of each axis until the anchors and g clear both labels by
    # CLEAR; the labels hang from the axes top, so only the data move
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    r_in = math.sqrt(POINT_S) / 2 / 72                     # marker radius, inches
    for (ax, s, _, _, px, py, _), pair in zip(placed, boxes):
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        fx = (px - x_lo) / (x_hi - x_lo)
        need = y_hi
        for t in pair:
            (bx0, by0), (bx1, _) = ax.transAxes.inverted().transform(
                t.get_bbox_patch().get_window_extent(rend).get_points())
            over = (fx > bx0 - r_in / colw) & (fx < bx1 + r_in / colw)
            room = by0 - (CLEAR + r_in) / AXH
            if over.any():
                need = max(need, y_lo + (py[over].max() - y_lo) / room)
        ax.set_ylim(y_lo, need)
    # one key for all columns
    kax = axes_in(fig, W, H, W0 - R_M + KEY_GAP, BELOW + (AXH - KEY) / 2, KEY, KEY)
    g = np.linspace(0.0, 1.0, 64)
    GX, GY = np.meshgrid(g, g)
    kax.imshow(position_colours(GX.ravel(), GY.ravel()).reshape(64, 64, 3),
               origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    kax.set_xticks([]); kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_visible(True); sp.set_color(C.C_MUTED); sp.set_linewidth(1.0)
    kax.set_xlabel(r"$\Pi_1\rightarrow$", fontsize=LABEL_FS, color=C.C_MUTED,
                   labelpad=1.5)
    kax.set_ylabel(r"$\Pi_2\rightarrow$", fontsize=LABEL_FS, color=C.C_MUTED,
                   labelpad=1.5)
    kax.text(0.5, 1.06, "Point colour", transform=kax.transAxes, ha="center",
             va="bottom", fontsize=LABEL_FS, color=C.C_TEXT, style="italic")
    kax.text(0.5, -0.50, "(all panels)", transform=kax.transAxes, ha="center",
             va="top", fontsize=LABEL_FS, color=C.C_MUTED, style="italic")
    # no anchor may sit under a label
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rad = math.sqrt(POINT_S) / 2 * fig.dpi / 72
    for (ax, s, _, _, _, _, _), pair in zip(placed, boxes):
        pts = ax.transData.transform(np.c_[s["u"], s["yn"]])
        hit = 0
        for t in pair:
            bb = t.get_bbox_patch().get_window_extent(rend)
            hit += int(np.sum((pts[:, 0] > bb.x0 - rad) & (pts[:, 0] < bb.x1 + rad)
                              & (pts[:, 1] > bb.y0 - rad) & (pts[:, 1] < bb.y1 + rad)))
        print("%-24s anchors under the labels: %d" % (s["name"], hit))
    save(fig, "fig3_stage2")
    C.compare_published(S)


if __name__ == "__main__":
    main()
