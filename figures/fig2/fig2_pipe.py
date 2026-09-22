# -*- coding: utf-8 -*-
"""Figure 2 -- rough-pipe flow: the results panels, one file each.

The author assembles the figure and already has panels a-c (the pipe render, the
dimensionless formulation, the three regimes).  This script draws the results:

  d  fig2d_stage1   stage one per window: the direction read at each of the 40
                    anchors, the spread of the readings about their mean, and
                    which stage-one condition of Figure 1c holds
  e  fig2e_stage2   stage two per window: lambda at the 40 anchor centres against
                    the single group u-hat = Re^a (eps/D)^b at the searched angle,
                    the fitted g, which stage-two condition of Figure 1c holds,
                    and under it the verdict of both stages

PANELS=d (default), PANELS=e or PANELS=d,e picks which to draw.  Each is exported
at its planned final size, so 7 pt text stays 7 pt if placed at 100 %.

Choices that differ from the author's earlier rough-pipe figure:
  * the probing-policy panel is gone: the DRL planner was replaced by 15
    equispaced angles, and the measurement is Figure 1's
  * the results carry the noise and the verdicts, which is what the paper adds
  * panel d is drawn in the style of the author's plot_three_regimes_combined.py
    (regime colours, tail-anchored arrows of one length, a shared angle axis and
    a shared density axis, a dashed mean and its label), with the current
    protocol's data instead of the DRL read-out
  * the read-out is axial (mod 180 deg), so every angle is unwrapped to within
    90 deg of its regime's mean before it is drawn: at low Re most readings sit
    near 180 deg and a few near 0 deg, and without the unwrap those few would
    point the other way and sit at the far end of the angle axis
  * each histogram carries stage one's outcome written as Figure 1c's condition,
    not as a verdict word: REDUCIBLE needs both stages, so the verdict comes after
    stage two.  The numbers behind it (sigma_alpha, sigma_noise, sigma_true, q95)
    go to a table in the paper, where the benchmarks can be read side by side
  * panel e is drawn in the style of the author's
    v4_transition/plot_three_regimes_collapse_row.py (one column per window, the
    single group under each title, points coloured by position in the window
    with that script's four corner colours, one key for all columns), with three
    differences: lambda is plotted, not log lambda, because stage two fits lambda
    itself (framework_pipeline.py fits the polynomial to yn = F0 + e0); u-hat is
    written from this realisation's (a, b); and "no common reduction" is not
    printed, since it is a verdict given before the verdict
  * e carries stage two's condition in a light box and, under it, the verdict of
    both stages in a dark box: Figure 1c defines REDUCIBLE, NOT-RED and
    UNRESOLVED only for both stages together
  * every line is at least 1 pt at final size, the thinnest Nature Communications
    allows (How to submit, General figure guidelines): axes, ticks, curves, the
    dashed mean, the rug, box edges and the arrow shafts
  * all text is 7 pt, the largest Nature's figure guide allows; the REDUCIBLE
    conditions break over two lines where Figure 1c's do, since at 7 pt stage
    one's does not fit a column on one line
Data and the stage-two computation: fig2_common.py.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C

ROOT = Path(__file__).parent
MM = 1 / 25.4
UNITS = ("u0", "u4", "u1")
NAMES = (r"Low $Re$", r"Medium $Re$", r"High $Re$")
PANELS = os.environ.get("PANELS", "d").split(",")
LABEL_FS = 7                     # every label at 7 pt, the ceiling of Nature's figure guide

d = C.load()
S = {u: C.stages(d, u) for u in UNITS}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), transparent=True)
    size = fig.get_size_inches() * 25.4
    plt.close(fig)
    print("%-14s %.0f x %.0f mm" % (name, size[0], size[1]))


def axes_in(fig, W, H, left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def pill(ax, x, y, verdict, ha="right", va="bottom", text=None):
    fc, ec, tc = C.VERDICT[verdict]
    return ax.text(x, y, verdict if text is None else text, transform=ax.transAxes,
                   ha=ha, va=va, fontsize=LABEL_FS, color=tc,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=fc, ec=ec, lw=1.0))


# ---- d: stage one, in the style of plot_three_regimes_combined.py --------------------
REGIME_COLOUR = ("#e377c2", "#d62728", "#1f77b4")         # low / medium / high Re
KDE_BW = 2.5                                              # degrees, as in that script
ARROW_IN = 0.12                                           # arrow length at final size
SHAFT_IN = 1.0 / 72                                       # 1 pt, NC's thinnest line
HEAD_IN = (0.042, 0.0525, 0.04725)       # head width, length, axis length (inches)
# stage one's outcome as Figure 1c writes its part of the conditions (CONDS in
# fig1c_flow_v2.py), not as a verdict: REDUCIBLE also needs stage two.  At 7 pt
# the REDUCIBLE one is wider than a column, so it breaks where Figure 1c's lines do
STAGE1_CONDITION = {
    "REDUCIBLE": r"$\hat\sigma_{\rm true}<\tau_1$" "\n" r"and $q_{95}<\tau_1$",
    "NOT-RED": r"$\hat\sigma_{\rm true}>\max(\tau_1,q_{95})$",
}


def unwrap_deg(ang_rad, centre_deg):
    """Axial angles (mod 180) moved to within 90 deg of the regime's mean."""
    a = np.degrees(np.asarray(ang_rad))
    return centre_deg + ((a - centre_deg + 90.0) % 180.0) - 90.0


def kde(samples, grid, bw):
    z = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * z ** 2).mean(1) / (bw * math.sqrt(2.0 * math.pi))


def panel_d():
    W = 95 * MM
    L_M, R_M, GAP = 0.38, 0.05, 0.28
    colw = (W - L_M - R_M - 2 * GAP) / 3
    TITLE, FIELD, BELOW1 = 0.17, 1.05, 0.32
    # the histograms run to YTOP ymax, not 1.35, so stage one's condition has its
    # own band above the mean labels; HIST grows with YTOP so the curves keep their
    # scale of 0.80 in per 1.45 ymax
    YTOP = 1.89
    HIST, BELOW2, XLAB = 0.80 * (YTOP + 0.10) / 1.45, 0.16, 0.16
    H = TITLE + FIELD + BELOW1 + HIST + BELOW2 + XLAB + 0.02
    y_hist = XLAB + BELOW2
    y_field = y_hist + HIST + BELOW1

    prep = []
    for u in UNITS:
        s = S[u]
        centre = math.degrees(s["abar"])
        prep.append(dict(s=s, centre=centre, ang=unwrap_deg(s["ang"], centre)))
    xlo = min(0.0, min(p["ang"].min() for p in prep)) - 6.0
    xhi = max(180.0, max(p["ang"].max() for p in prep)) + 6.0
    grid = np.linspace(xlo, xhi, 1600)
    for p in prep:
        p["dens"] = kde(p["ang"], grid, KDE_BW)
    ymax = max(p["dens"].max() for p in prep)              # one density axis

    fig = plt.figure(figsize=(W, H))
    for j, (p, name, col) in enumerate(zip(prep, NAMES, REGIME_COLOUR)):
        s, left = p["s"], L_M + j * (colw + GAP)

        # the direction read at each anchor
        ax = axes_in(fig, W, H, left, y_field, colw, FIELD)
        xs, ys = s["CX"], s["CY"]
        rad = np.radians(p["ang"])
        ax.scatter(xs, ys, s=2, color=col, alpha=0.55, edgecolor="none", zorder=4)
        # a 1 pt shaft; arrow and head are 1.4 times their first size (0.085 in
        # long, head 4, 5 and 4.5 widths of a 0.0075 in shaft), set in inches
        ax.quiver(xs, ys, np.cos(rad), np.sin(rad), color=col, alpha=0.9,
                  angles="uv", pivot="tail", units="inches", scale_units="inches",
                  scale=1.0 / ARROW_IN, width=SHAFT_IN,
                  headwidth=HEAD_IN[0] / SHAFT_IN, headlength=HEAD_IN[1] / SHAFT_IN,
                  headaxislength=HEAD_IN[2] / SHAFT_IN, zorder=3)
        xspan, yspan = np.ptp(xs), np.ptp(ys)
        ax.set_xlim(xs.min() - 0.20 * xspan, xs.max() + 0.20 * xspan)
        ax.set_ylim(ys.min() - 0.23 * yspan, ys.max() + 0.23 * yspan)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(direction="in", length=2)
        ax.set_xlabel(r"$\ln Re$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$\ln(\varepsilon/D)$", labelpad=1)
        ax.set_title(name, fontsize=7, fontweight="bold", color=col, pad=3)

        # the readings' distribution, on one angle axis and one density axis
        axh = axes_in(fig, W, H, left, y_hist, colw, HIST)
        dens = p["dens"]
        axh.fill_between(grid, dens, color=col, alpha=0.80, linewidth=0, zorder=3)
        axh.plot(grid, dens, color=col, lw=1.0, zorder=4)
        axh.vlines(p["ang"], -0.085 * ymax, -0.02 * ymax, color=col, lw=1.0,
                   alpha=0.7, zorder=4)
        # the dashed mean stops at the curves' band, so the numbers above it are
        # not struck through
        axh.vlines(p["centre"], 0, 1.08 * ymax, linestyles="--", lw=1.0,
                   color=col, alpha=0.55, zorder=2)
        # the mean's label beside its line, on the side with room
        right = p["centre"] < 130.0
        axh.text(p["centre"] + (4 if right else -4), 1.10 * ymax,
                 r"$\bar\alpha \approx %.0f$°" % (p["centre"] % 180.0),
                 ha="left" if right else "right", va="bottom", fontsize=LABEL_FS,
                 color=col, style="italic", fontweight="bold", zorder=5)
        # stage one's outcome, written as its condition, in the top-left corner
        # above the mean labels (at their height a label this wide ran into them
        # at low and high Re).  The numbers behind it (sigma_alpha, sigma_noise,
        # sigma_true, q95) go to a table in the paper
        pill(axh, 0.06, 0.97, s["c1"], ha="left", va="top",
             text=STAGE1_CONDITION[s["c1"]])
        axh.set_xlim(xlo, xhi); axh.set_ylim(-0.10 * ymax, YTOP * ymax)
        # three ticks: five crowded each other in a 23 mm column
        axh.set_xticks([0, 90, 180])
        axh.set_xticklabels(["0°", "90°", "180°"])
        axh.set_yticks([0]); axh.set_yticklabels(["0"])
        axh.tick_params(direction="in", length=2)
        if j == 0:
            axh.set_ylabel("Density", labelpad=1)
    fig.text((L_M + 1.5 * colw + GAP) / W, 0.02 / H,
             r"Recovered direction angle $\alpha$", ha="center", va="bottom",
             fontsize=7, color=C.C_TEXT)
    save(fig, "fig2d_stage1")


# ---- e: stage two -------------------------------------------------------------------------
# in the style of the author's v4_transition/plot_three_regimes_collapse_row.py
POS_CORNERS = ("#eef1f5", "#2e8fdf", "#f15a29", "#8f53a8")   # its C00, C10, C01, C11
POINT_S = 8                                                  # marker area, pt^2
# stage two's part of Figure 1c's conditions, as STAGE1_CONDITION is stage one's.
# Stage two compares rho_hat_true = sqrt(max(rho^2 - tau_min^2, 0)), not rho (v2 in
# framework_pipeline.py; C.stages computes c2 from it), hatted like sigma_hat_true;
# both boxes said rho until 2026-09-15, as Figure 1c did
STAGE2_CONDITION = {
    "REDUCIBLE": r"$\hat\rho_{\rm true}<\tau$" "\n" r"and $\tau_{\min}<\tau$",
    "NOT-RED": r"$\hat\rho_{\rm true}>\max(\tau,\tau_{\min})$",
}


def position_colours(v_re, v_ep):
    """That script's bivariate_colors: bilinear between the four corner colours."""
    a = [np.array(to_rgb(c)) for c in POS_CORNERS]
    vr, ve = np.asarray(v_re)[:, None], np.asarray(v_ep)[:, None]
    return np.clip((1 - vr) * (1 - ve) * a[0] + vr * (1 - ve) * a[1]
                   + (1 - vr) * ve * a[2] + vr * ve * a[3], 0, 1)


def group_label(a, b):
    """u-hat = Re^a (eps/D)^b for the subtitle, exponents to two decimals."""
    def power(base, p, paren):
        if abs(p) < 0.005:
            return ""
        if abs(p - 1.0) < 0.005:
            return base
        e = "%d" % round(p) if abs(p - round(p)) < 0.005 else "%.2f" % p
        return (r"(%s)^{%s}" if paren else r"%s^{%s}") % (base, e)
    return r"$\hat{u}\approx %s%s$" % (power("Re", a, False),
                                        power(r"\varepsilon/D", b, True))


def solid_pill(ax, x, y, verdict, ha="left", va="top"):
    """The verdict of both stages: dark fill and white text, unlike a condition."""
    dark = C.VERDICT[verdict][2]
    return ax.text(x, y, verdict, transform=ax.transAxes, ha=ha, va=va, fontsize=LABEL_FS,
                   color="white", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                                            fc=dark, ec=dark, lw=1.0))


def panel_e():
    W0 = 95 * MM                                     # panel d's width: same columns
    L_M, R_M, GAP = 0.38, 0.05, 0.28
    colw = (W0 - L_M - R_M - 2 * GAP) / 3
    KEY_GAP, KEY, KEY_R = 0.24, 0.30, 0.18           # the one key, right of the columns
    W = W0 - R_M + KEY_GAP + KEY + KEY_R
    BELOW, AXH, TITLE = 0.32, 1.15, 0.33
    H = BELOW + AXH + TITLE
    # condition box to verdict box, 1 mm as seen: the patch extents leave out the
    # 1 pt edges, half of it on each box
    BOX_GAP = (1.0 * MM + 1.0 / 72) / AXH
    PAD = 0.22 * LABEL_FS / 72 / AXH                 # a box's pad, in axes units
    CLEAR = 1.0 * MM                                 # labels to the nearest data, inches
    # drawn at the saving dpi, so the label sizes measured below are the saved ones
    # (at 100 dpi the two-line labels came out taller and the gaps uneven)
    fig = plt.figure(figsize=(W, H), dpi=400)
    placed = []
    for j, (u, name, col) in enumerate(zip(UNITS, NAMES, REGIME_COLOUR)):
        s, left = S[u], L_M + j * (colw + GAP)
        ax = axes_in(fig, W, H, left, BELOW, colw, AXH)
        uu, yy = s["u"], s["yn"]
        # stage two's fitted g under the anchors, which are coloured by where they
        # sit in the window; no white rim on the points as that script had, since
        # a 0.2 pt rim is a line thinner than NC's 1 pt
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
        # steps of 1, 2 or 5 only: at 7 pt a label like 0.042 runs up against the
        # column to its left
        ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.tick_params(direction="in", length=2)
        ax.set_xlabel(r"$\ln\hat{u}$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$\lambda$", labelpad=1)
        # title as in panel d, the single group under it
        cx = (left + colw / 2) / W
        fig.text(cx, (BELOW + AXH + 0.235) / H, name, ha="center", va="baseline",
                 fontsize=7, fontweight="bold", color=col)
        fig.text(cx, (BELOW + AXH + 0.07) / H,
                 group_label(math.cos(s["a_star"]), math.sin(s["a_star"])),
                 ha="center", va="baseline", fontsize=LABEL_FS, color=C.C_MUTED)
        c2 = C.STAGE2_NAME[s["c2"]]
        placed.append((ax, s, np.r_[uu, ug], np.r_[yy, gg],
                       pill(ax, 0.06, 0.97, c2, ha="left", va="top",
                            text=STAGE2_CONDITION[c2])))
    # the verdict of both stages, right under stage two's condition
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax, s, _, _, cond in placed:
        y_low = ax.transAxes.inverted().transform(
            (0.0, cond.get_bbox_patch().get_window_extent(rend).y0))[1]
        boxes.append((cond, solid_pill(ax, 0.06, y_low - BOX_GAP - PAD,
                                       C.JOINT_NAME[s["jt"]])))
    # raise the top of each axis until the anchors and g clear both labels by
    # CLEAR; the labels hang from the axes top, so only the data move
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    r_in = math.sqrt(POINT_S) / 2 / 72                     # marker radius, inches
    for (ax, s, px, py, _), pair in zip(placed, boxes):
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
    # one key for all columns, as in that script
    kax = axes_in(fig, W, H, W0 - R_M + KEY_GAP, BELOW + (AXH - KEY) / 2, KEY, KEY)
    g = np.linspace(0.0, 1.0, 64)
    GX, GY = np.meshgrid(g, g)
    kax.imshow(position_colours(GX.ravel(), GY.ravel()).reshape(64, 64, 3),
               origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    kax.set_xticks([]); kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_visible(True); sp.set_color(C.C_MUTED); sp.set_linewidth(1.0)
    kax.set_xlabel(r"$Re\rightarrow$", fontsize=LABEL_FS, color=C.C_MUTED, labelpad=1.5)
    kax.set_ylabel(r"$\varepsilon/D\rightarrow$", fontsize=LABEL_FS, color=C.C_MUTED,
                   labelpad=1.5)
    kax.text(0.5, 1.06, "Point colour", transform=kax.transAxes, ha="center",
             va="bottom", fontsize=LABEL_FS, color=C.C_TEXT, style="italic")
    kax.text(0.5, -0.50, "(all panels)", transform=kax.transAxes, ha="center",
             va="top", fontsize=LABEL_FS, color=C.C_MUTED, style="italic")
    # no anchor may sit under a label
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rad = math.sqrt(POINT_S) / 2 * fig.dpi / 72
    for (ax, s, _, _, _), pair in zip(placed, boxes):
        pts = ax.transData.transform(np.c_[s["u"], s["yn"]])
        hit = 0
        for t in pair:
            bb = t.get_bbox_patch().get_window_extent(rend)
            hit += int(np.sum((pts[:, 0] > bb.x0 - rad) & (pts[:, 0] < bb.x1 + rad)
                              & (pts[:, 1] > bb.y0 - rad) & (pts[:, 1] < bb.y1 + rad)))
        print("%-24s anchors under the labels: %d" % (s["name"], hit))
    save(fig, "fig2e_stage2")


for p in PANELS:
    {"d": panel_d, "e": panel_e}[p.strip()]()
C.compare_published(S)
