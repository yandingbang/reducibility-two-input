# -*- coding: utf-8 -*-
"""Figure 4 -- the two exact reductions, microfluidic droplets and column buckling,
drawn as one figure.

Layout from the author's plot_grid_single.py (this file began as a copy of it):
one column per system under the system's name, rows named at the left, panels
a-h, and that script's two schematics.

Changed, to match Figures 1-3 and Nature Communications' figure rules:
  * a white background: no row bands and no rounded container (NC asks for
    display items on a white background, without unnecessary colour or
    boxing); thin light rules divide the rows
  * drawn at final size, 180 mm wide; Arial 7 pt with upright Times New Roman
    Greek (Figure2/fig2_common.py); panel letters 8 pt bold without a period;
    every line at least 1 pt
  * the data are one realisation of the current protocol
    (deploy_stop/figures/fig2_method_data.npz), not the DRL read-out: the fields
    are the directions read at the 40 anchors, and the collapse plots those 40
    anchors with stage two's fitted g instead of 140/150 random samples of the law
  * stage one carries the angle distribution with the exact direction and stage
    one's condition; stage two carries stage two's condition and, under it, the
    verdict of both stages (as in Figures 2 and 3)
  * the group row sets the recovered group beside the exact one, both written as
    unit exponent vectors: Ca^-1 phi^(1/3) normalised is Ca^-0.95 phi^0.32, and a
    power of a group is the same group
  * the response itself is plotted against ln u-hat, since stage two fits it;
    natural logarithms throughout
The exact laws: microfluidic L_d/w depends on Ca^-1 phi^(1/3) only (resp_micro;
a constructed monomial, the phi dependence after Thorsen et al. 2001); buckling
sigma_cr/sigma_y = pi^2 / (pi^2 + (kappa L/r)^2 sigma_y/E), the Rankine-Gordon
formula (resp_buck).

Output: fig4.png / .svg
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, data, stages

MM = 1 / 25.4
FS = 7                           # every label at 7 pt, the ceiling of Nature's figure guide
TXT, MUT = C.C_TEXT, C.C_MUTED
C_MICRO, C_BUCK = "#3f80bf", "#42627e"                   # the author's system colours
RULE = "#d5d9de"                                          # the rules between rows
UNITS = ("u2", "u3")
NAMES = ("Microfluidic droplet generation", "Column buckling")
COLS = (C_MICRO, C_BUCK)
ROW_NAMES = ("Physical\nsystems", "Stage one:\ndirections", "Recovered\ngroups",
             "Stage two:\ncollapse")
FIELD_LABELS = ((r"$\ln Ca$", r"$\ln\varphi$"),
                (r"$\ln(\kappa L/r)$", r"$\ln(\sigma_y/E)$"))
BASES = ((r"Ca", r"\varphi"), (r"(\kappa L/r)", r"(\sigma_y/E)"))
RESPONSE = (r"$L_d/w$", r"$\sigma_{cr}/\sigma_y$")
RESPONSE_DEF = (r"$\Pi_o = L_d/w$", r"$\Pi_o = \sigma_{cr}/\sigma_y$")
# the exponents of each law's one group on its (x, y) axes
EXACT = {"u2": (-1.0, 1.0 / 3.0),                         # Ca^-1 phi^(1/3)
         "u3": (2.0, 1.0)}                                # (kappa L/r)^2 sigma_y/E
# Figure 1c's conditions, one line each: these cells are wide enough
STAGE1_CONDITION = {
    "REDUCIBLE": r"$\hat\sigma_{\rm true}<\tau_1$ and $q_{95}<\tau_1$",
    "NOT-RED": r"$\hat\sigma_{\rm true}>\max(\tau_1,q_{95})$",
}
# Stage two compares rho_hat_true = sqrt(max(rho^2 - tau_min^2, 0)), not rho (v2 in
# framework_pipeline.py; C.stages computes c2 from it), hatted like sigma_hat_true;
# both said rho until 2026-09-15, as Figures 1c, 2e and 3 did
STAGE2_CONDITION = {
    "REDUCIBLE": r"$\hat\rho_{\rm true}<\tau$ and $\tau_{\min}<\tau$",
    "NOT-RED": r"$\hat\rho_{\rm true}>\max(\tau,\tau_{\min})$",
}
KDE_BW = 2.5                                                # degrees, as in Figure 2
ARROW_IN = 0.12                                             # arrow length at final size
SHAFT_IN = 1.0 / 72                                         # 1 pt, NC's thinnest line
HEAD_IN = (0.042, 0.0525, 0.04725)       # head width, length, axis length (inches)
EDGE_IN = 0.06                           # outermost anchor or arrow tip to an axis edge
POINT_S = 8                                                 # marker area, pt^2
BUCK_YMAX = 6.35                         # the cantilever drawing's height, data units

d = C.load()
S = {u: C.stages(d, u) for u in UNITS}


def unwrap_deg(ang_rad, centre_deg):
    """Axial angles (mod 180) moved to within 90 deg of the window's mean."""
    a = np.degrees(np.asarray(ang_rad))
    return centre_deg + ((a - centre_deg + 90.0) % 180.0) - 90.0


def kde(samples, grid, bw):
    z = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * z ** 2).mean(1) / (bw * math.sqrt(2.0 * math.pi))


def extent(scale, pos, lead):
    """(low, high) in inches of anchors at pos (ln units) whose arrows reach lead
    inches along this axis, with EDGE_IN of room on both sides."""
    return (float(np.min(pos * scale + np.minimum(lead, 0.0))) - EDGE_IN,
            float(np.max(pos * scale + np.maximum(lead, 0.0))) + EDGE_IN)


def solve_scale(total_in, pos, lead):
    """Inches per ln unit at which the extent is total_in."""
    lo, hi = 1e-3, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a, b = extent(mid, pos, lead)
        lo, hi = (mid, hi) if b - a < total_in else (lo, mid)
    return 0.5 * (lo + hi)


def unit_group(a, b, bases):
    """Pi1^a Pi2^b with (a, b) a unit exponent vector, exponents to two decimals."""
    return r"\,".join(r"%s^{%.2f}" % (base, p) for base, p in zip(bases, (a, b))
                      if abs(p) >= 0.005)


def condition_box(ax, x, y, verdict, text, ha):
    fc, ec, tc = C.VERDICT[verdict]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top",
                   multialignment="left", fontsize=FS, color=tc,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=fc, ec=ec, lw=1.0))


def verdict_box(ax, x, y, verdict, ha):
    """The verdict of both stages: dark fill and white text, unlike a condition."""
    dark = C.VERDICT[verdict][2]
    return ax.text(x, y, verdict, transform=ax.transAxes, ha=ha, va="top",
                   fontsize=FS, color="white",
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25",
                             fc=dark, ec=dark, lw=1.0))


def draw_micro(ax):
    """The T-junction of the author's grid, at 7 pt with 1 pt lines; square data units."""
    ax.set_xlim(0, 9.75); ax.set_ylim(0, 5.0); ax.axis("off")
    OIL, WHILT, WALL = "#f4ecdf", "#bcd9f0", "#3a3f47"
    MX0, MX1, MY0, MY1 = 0.8, 8.95, 1.15, 2.45
    AL = 0.65                                  # one arrow length for Q_c, Q_d and the outlet
    SX0, SX1, SY1 = 2.65, 3.65, 4.1
    nx = 0.5 * (SX0 + SX1)
    outline = [(MX0, MY0), (MX1, MY0), (MX1, MY1), (SX1, MY1),
               (SX1, SY1), (SX0, SY1), (SX0, MY1), (MX0, MY1)]
    ax.add_patch(Polygon(outline, closed=True, facecolor=OIL, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(outline, closed=True, facecolor="none", edgecolor=WALL,
                         lw=1.6, joinstyle="round", zorder=4))

    def droplet(cx, cy, w, h):
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=C_MICRO, edgecolor=WALL, lw=1.0,
                             zorder=5))
        ax.add_patch(Ellipse((cx - 0.22 * w, cy + 0.20 * h), 0.42 * w, 0.30 * h,
                             facecolor=WHILT, edgecolor="none", alpha=0.72, zorder=6))

    # the dispersed phase's channel, the neck and the forming droplet
    ax.add_patch(Rectangle((SX0, MY1), SX1 - SX0, SY1 - MY1, facecolor=C_MICRO,
                           edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((nx - 0.12, SY1 - 0.7), 0.42, 0.5, facecolor=WHILT,
                         edgecolor="none", alpha=0.55, zorder=3))
    ax.add_patch(Polygon([(nx - 0.25, MY1 + 0.02), (nx + 0.25, MY1 + 0.02),
                          (nx + 0.13, 2.08), (nx - 0.13, 2.08)], closed=True,
                         facecolor=C_MICRO, edgecolor="none", zorder=2))
    droplet(nx, 1.72, 1.0, 0.88)
    for cx in (5.0, 6.35, 7.7):                # the droplet train downstream
        droplet(cx, 1.8, 1.05, 0.92)
    arrow = dict(arrowstyle="-|>", lw=1.6, color=WALL)
    ax.annotate("", xy=(MX0, 1.8), xytext=(MX0 - AL, 1.8), arrowprops=arrow)
    ax.text(MX0 - AL - 0.06, 2.18, r"$Q_c$", fontsize=FS, color=TXT, va="bottom")
    ax.annotate("", xy=(nx, SY1 + 0.12), xytext=(nx, SY1 + 0.12 + AL), arrowprops=arrow)
    ax.text(nx + 0.28, SY1 + 0.12 + AL * 0.6, r"$Q_d$", fontsize=FS, color=TXT,
            ha="left", va="center")
    ax.annotate("", xy=(MX1 + AL, 1.8), xytext=(MX1, 1.8), arrowprops=arrow)
    dim = dict(arrowstyle="<->", lw=1.0, color=MUT)
    ax.annotate("", xy=(1.35, MY0), xytext=(1.35, MY1), arrowprops=dim)
    ax.text(1.55, 1.8, r"$w$", fontsize=FS, color=TXT, va="center")
    ax.annotate("", xy=(6.35 - 0.62, 2.72), xytext=(6.35 + 0.62, 2.72), arrowprops=dim)
    ax.text(6.35, 2.86, r"$L_d$", fontsize=FS, color=TXT, ha="center", va="bottom")


def draw_buckling(ax):
    """The cantilever of the author's grid, at 7 pt with 1 pt lines; square data units."""
    ax.set_xlim(0, 10); ax.set_ylim(0, BUCK_YMAX); ax.axis("off")
    FILL, EDGE = "#7e93ae", "#33497a"
    yb0, yt, A, hw = 0.8, 5.2, 1.0, 0.16
    yy = np.linspace(yb0, yt, 50)
    xc = 3.6 + A * (1 - np.cos(np.pi / 2 * (yy - yb0) / (yt - yb0)))
    body = np.vstack([np.column_stack([xc - hw, yy]), np.column_stack([xc + hw, yy])[::-1]])
    ax.add_patch(Polygon(body, closed=True, facecolor=FILL, edgecolor=EDGE, lw=1.4))
    ax.add_patch(Rectangle((3.6 - 0.55, yb0 - 0.32), 1.1, 0.32, facecolor="#e2e7ec",
                           edgecolor=EDGE, lw=1.2))
    for xx in np.linspace(3.6 - 0.55, 3.6 + 0.55, 6):     # the clamped base's hatching
        ax.plot([xx, xx - 0.13], [yb0 - 0.32, yb0 - 0.5], color=MUT, lw=1.0)
    # the load: a longer shaft and a smaller head than the grid's, whose head made up
    # the whole arrow at this size
    ax.add_patch(FancyArrowPatch((3.6 + A, yt + 1.05), (3.6 + A, yt + 0.12),
                                 arrowstyle="-|>", mutation_scale=8, lw=1.4, color=TXT))
    ax.text(3.6 + A + 0.25, yt + 0.6, r"$P$", fontsize=FS, color=TXT)
    ax.annotate("", xy=(6.0, yb0), xytext=(6.0, yt),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=MUT))
    ax.text(6.2, (yb0 + yt) / 2, r"$L$", fontsize=FS, color=TXT, va="center")
    ax.add_patch(Circle((8.2, 3.0), 0.62, facecolor=FILL, edgecolor=EDGE, lw=1.2))
    ax.add_patch(Circle((8.2, 3.0), 0.31, fill=False, edgecolor=C_BUCK, lw=1.0,
                        ls=(0, (2, 2))))
    ax.text(8.2, 2.0, r"$r$, $\sigma_y$, $E$", fontsize=FS, color=TXT, ha="center")


def main():
    W = 180 * MM
    LAB_W, CGAP, R_M = 0.80, 0.22, 0.04        # row-name column, gap between systems, right
    colW = (W - LAB_W - CGAP - R_M) / 2
    COL_X = (LAB_W, LAB_W + colW + CGAP)
    TOP, TITLE, RULE_GAP, BOTTOM = 0.03, 0.18, 0.07, 0.03
    ROW_H = (1.30, 1.60, 0.56, 1.62)
    row_top, t = [], TOP + TITLE
    for h in ROW_H:
        row_top.append(t)
        t += h + 2 * RULE_GAP
    H = t - 2 * RULE_GAP + BOTTOM
    # drawn at the saving dpi, so the label sizes measured below are the saved ones
    fig = plt.figure(figsize=(W, H), dpi=400)

    def axes(left, top, width, height):        # inches; top measured from the figure top
        return fig.add_axes([left / W, (H - top - height) / H, width / W, height / H])

    def text(x, top, s, **kw):
        return fig.text(x / W, (H - top) / H, s, **kw)

    def letter(col, row, s):
        text(COL_X[col], row_top[row] + 0.01, s, ha="left", va="top", fontsize=8,
             fontweight="bold", color=TXT)

    # ---- the frame: system names, row names, rules between the rows ----------------
    for x0, name in zip(COL_X, NAMES):
        text(x0 + colW / 2, TOP + TITLE / 2, name, ha="center", va="center", fontsize=FS,
             fontweight="bold", color=TXT)
    for i, (name, rt, h) in enumerate(zip(ROW_NAMES, row_top, ROW_H)):
        text(LAB_W / 2 - 0.03, rt + h / 2, name, ha="center", va="center", fontsize=FS,
             fontweight="bold", color=TXT, linespacing=1.3)
        if i:
            y = (H - (rt - RULE_GAP)) / H
            fig.add_artist(Line2D([0.03 / W, (W - R_M) / W], [y, y], color=RULE, lw=1.0))

    # ---- a, b: the physical systems -------------------------------------------------
    h1 = ROW_H[0] - 0.08
    draw_micro(axes(COL_X[0] + (colW - 1.95 * h1) / 2, row_top[0] + 0.04, 1.95 * h1, h1))
    wb = h1 * 10 / BUCK_YMAX
    draw_buckling(axes(COL_X[1] + (colW - wb) / 2, row_top[0] + 0.04, wb, h1))
    letter(0, 0, "a"); letter(1, 0, "b")

    # ---- c, d: stage one, the directions read and their distribution ----------------
    F_L, F_W, H_GAP = 0.42, 1.00, 0.44         # field: left room, width; field to histogram
    A_TOP, A_BOT = 0.16, 0.34                  # room for the letter above, ticks below
    AH = ROW_H[1] - A_TOP - A_BOT
    HW = colW - F_L - F_W - H_GAP - 0.02
    prep = []
    for u in UNITS:
        s = S[u]
        centre = math.degrees(s["abar"])
        ang = unwrap_deg(s["ang"], centre)
        rad = np.radians(ang)
        ea, eb = EXACT[u]
        prep.append(dict(s=s, centre=centre, ang=ang,
                         exact=float(unwrap_deg(math.atan2(eb, ea), centre)),
                         dx=ARROW_IN * np.cos(rad), dy=ARROW_IN * np.sin(rad)))
    xlo = min(0.0, min(p["ang"].min() for p in prep)) - 6.0
    xhi = max(180.0, max(p["ang"].max() for p in prep)) + 6.0
    grid = np.linspace(xlo, xhi, 1600)
    for p in prep:
        p["dens"] = kde(p["ang"], grid, KDE_BW)
    ymax = max(p["dens"].max() for p in prep)              # one density axis
    # the histograms run to YTOP ymax: the condition box on top, the exact and mean
    # labels stacked under it, the spike below
    YTOP = 2.0
    per_ymax = AH / (YTOP + 0.10)                          # inches per ymax
    LAB_IN = 0.10                                          # one label's height
    y_mean = 1.10
    y_exact = y_mean + (LAB_IN + 0.02) / per_ymax
    for j, (p, col, (xl, yl)) in enumerate(zip(prep, COLS, FIELD_LABELS)):
        s = p["s"]
        top = row_top[1] + A_TOP
        ax = axes(COL_X[j] + F_L, top, F_W, AH)
        sx = solve_scale(F_W, s["CX"], p["dx"])
        sy = solve_scale(AH, s["CY"], p["dy"])
        x0, x1 = extent(sx, s["CX"], p["dx"])
        y0, y1 = extent(sy, s["CY"], p["dy"])
        rad = np.radians(p["ang"])
        ax.scatter(s["CX"], s["CY"], s=2, color=col, alpha=0.55, edgecolor="none",
                   zorder=4)
        ax.quiver(s["CX"], s["CY"], np.cos(rad), np.sin(rad), color=col, alpha=0.9,
                  angles="uv", pivot="tail", units="inches", scale_units="inches",
                  scale=1.0 / ARROW_IN, width=SHAFT_IN,
                  headwidth=HEAD_IN[0] / SHAFT_IN, headlength=HEAD_IN[1] / SHAFT_IN,
                  headaxislength=HEAD_IN[2] / SHAFT_IN, zorder=3)
        ax.set_xlim(x0 / sx, x1 / sx)
        ax.set_ylim(y0 / sy, y1 / sy)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(direction="in", length=2)
        ax.set_xlabel(xl, labelpad=1)
        ax.set_ylabel(yl, labelpad=1)

        axh = axes(COL_X[j] + F_L + F_W + H_GAP, top, HW, AH)
        axh.fill_between(grid, p["dens"], color=col, alpha=0.80, linewidth=0, zorder=3)
        axh.plot(grid, p["dens"], color=col, lw=1.0, zorder=4)
        axh.vlines(p["ang"], -0.085 * ymax, -0.02 * ymax, color=col, lw=1.0,
                   alpha=0.7, zorder=4)
        axh.vlines(p["centre"], 0, 1.08 * ymax, linestyles="--", lw=1.0,
                   color=col, alpha=0.55, zorder=2)
        # the exact direction: a black stub above the spike, clear of the dashed mean
        axh.vlines(p["exact"], 1.14 * ymax, (y_exact + LAB_IN / per_ymax) * ymax, lw=1.0,
                   color=TXT, zorder=5)
        off, ha = (-4.0, "right") if p["centre"] > 90.0 else (4.0, "left")
        axh.text(p["centre"] + off, y_mean * ymax,
                 r"$\bar\alpha \approx %.1f$°" % (p["centre"] % 180.0), ha=ha,
                 va="bottom", fontsize=FS, color=col, style="italic", fontweight="bold",
                 zorder=5)
        axh.text(p["exact"] + off, y_exact * ymax, "exact %.1f°" % (p["exact"] % 180.0),
                 ha=ha, va="bottom", fontsize=FS, color=TXT, zorder=5)
        condition_box(axh, 0.06, 0.97, s["c1"], STAGE1_CONDITION[s["c1"]], "left")
        axh.set_xlim(xlo, xhi)
        axh.set_ylim(-0.10 * ymax, YTOP * ymax)
        axh.set_xticks([0, 90, 180])
        axh.set_xticklabels(["0°", "90°", "180°"])
        axh.set_yticks([0])
        axh.set_yticklabels(["0"])
        axh.tick_params(direction="in", length=2)
        axh.set_xlabel(r"Direction angle $\alpha$", labelpad=1)
        axh.set_ylabel("Density", labelpad=1)
    letter(0, 1, "c"); letter(1, 1, "d")

    # ---- e, f: the recovered group beside the exact one, as unit exponent vectors ----
    for j, u in enumerate(UNITS):
        s = S[u]
        ar, br = math.cos(s["a_star"]), math.sin(s["a_star"])
        ea, eb = EXACT[u]
        n = math.hypot(ea, eb)
        ea, eb = ea / n, eb / n
        if ea * ar + eb * br < 0:              # an axial direction: pick the recovered sign
            ea, eb = -ea, -eb
        lines = (("recovered", r"$\hat{u}\approx %s$" % unit_group(ar, br, BASES[j]), COLS[j]),
                 ("exact", r"$\hat{u}\approx %s$" % unit_group(ea, eb, BASES[j]), TXT),
                 ("response", RESPONSE_DEF[j], TXT))
        split = COL_X[j] + colW / 2 - 0.35
        for i, (word, formula, colour) in enumerate(lines):
            y = row_top[2] + 0.12 + i * 0.16
            text(split - 0.05, y, word, ha="right", va="center", fontsize=FS, color=MUT)
            text(split + 0.05, y, formula, ha="left", va="center", fontsize=FS, color=colour)
    letter(0, 2, "e"); letter(1, 2, "f")

    # ---- g, h: stage two, the response against the recovered group, and the verdict ---
    C_L, C_W, C_TOP, C_BOT = 0.62, 2.10, 0.14, 0.36
    CH = ROW_H[3] - C_TOP - C_BOT
    # condition box to verdict box, 1 mm as seen: the patch extents leave out the
    # 1 pt edges, half of it on each box
    BOX_GAP = (1.0 * MM + 1.0 / 72) / CH
    PAD = 0.22 * FS / 72 / CH                  # a box's pad, in axes units
    CLEAR = 1.0 * MM                           # labels to the nearest data, inches
    INSET = 0.06
    placed = []
    for j, (u, col, resp) in enumerate(zip(UNITS, COLS, RESPONSE)):
        s = S[u]
        ax = axes(COL_X[j] + C_L, row_top[3] + C_TOP, C_W, CH)
        uu, yy = s["u"], s["yn"]
        m, sd = uu.mean(), uu.std()
        coef = np.polyfit((uu - m) / sd, yy, s["deg"])
        ug = np.linspace(uu.min(), uu.max(), 300)
        gg = np.polyval(coef, (ug - m) / sd)
        ax.plot(ug, gg, color=MUT, lw=1.0, zorder=2)
        ax.scatter(uu, yy, s=POINT_S, color=col, edgecolor="none", zorder=3)
        xspan, yspan = np.ptp(uu), np.ptp(yy)
        ax.set_xlim(uu.min() - 0.06 * xspan, uu.max() + 0.06 * xspan)
        ax.set_ylim(yy.min() - 0.08 * yspan, yy.max() + 0.10 * yspan)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.tick_params(direction="in", length=2)
        ax.set_xlabel(r"$\ln\hat{u}$", labelpad=1)
        ax.set_ylabel(resp, labelpad=1)
        # the labels hang from the top corner away from the data
        rising = np.corrcoef(uu, yy)[0, 1] > 0
        bx, ha = (INSET, "left") if rising else (1.0 - INSET, "right")
        c2 = C.STAGE2_NAME[s["c2"]]
        placed.append((ax, s, bx, ha, np.r_[uu, ug], np.r_[yy, gg],
                       condition_box(ax, bx, 0.97, c2, STAGE2_CONDITION[c2], ha)))
    letter(0, 3, "g"); letter(1, 3, "h")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax, s, bx, ha, _, _, cond in placed:
        y_low = ax.transAxes.inverted().transform(
            (0.0, cond.get_bbox_patch().get_window_extent(rend).y0))[1]
        boxes.append((cond, verdict_box(ax, bx, y_low - BOX_GAP - PAD,
                                        C.JOINT_NAME[s["jt"]], ha)))
    # raise the top of each axis until the anchors and g clear both labels by CLEAR
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    r_in = math.sqrt(POINT_S) / 2 / 72                     # marker radius, inches
    for (ax, s, _, _, px, py, _), pair in zip(placed, boxes):
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        fx = (px - x_lo) / (x_hi - x_lo)
        need = y_hi
        for t_box in pair:
            (bx0, by0), (bx1, _) = ax.transAxes.inverted().transform(
                t_box.get_bbox_patch().get_window_extent(rend).get_points())
            over = (fx > bx0 - r_in / C_W) & (fx < bx1 + r_in / C_W)
            room = by0 - (CLEAR + r_in) / CH
            if over.any():
                need = max(need, y_lo + (py[over].max() - y_lo) / room)
        ax.set_ylim(y_lo, need)
    # no anchor may sit under a label
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rad = math.sqrt(POINT_S) / 2 * fig.dpi / 72
    for (ax, s, _, _, _, _, _), pair in zip(placed, boxes):
        pts = ax.transData.transform(np.c_[s["u"], s["yn"]])
        hit = 0
        for t_box in pair:
            bb = t_box.get_bbox_patch().get_window_extent(rend)
            hit += int(np.sum((pts[:, 0] > bb.x0 - rad) & (pts[:, 0] < bb.x1 + rad)
                              & (pts[:, 1] > bb.y0 - rad) & (pts[:, 1] < bb.y1 + rad)))
        print("%-18s anchors under the stage-two labels: %d" % (s["name"], hit))

    fig.savefig(ROOT / "fig4.png", dpi=400, facecolor="white")
    fig.savefig(ROOT / "fig4.svg", facecolor="white")
    plt.close(fig)
    print("fig4  %.0f x %.0f mm" % (W * 25.4, H * 25.4))
    for p in prep:
        print("%-18s mean %.2f deg, exact %.2f deg" % (p["s"]["name"], p["centre"] % 180,
                                                      p["exact"] % 180))
    C.compare_published(S)


if __name__ == "__main__":
    main()
