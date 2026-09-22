# -*- coding: utf-8 -*-
"""Figure 1c, version 2 -- the sketch's idea in Nature Communications style.

The idea taken from the sketch.  One measurement is read for its SIGNAL
(directions, collapse) and for its NOISE (the residual pool); the verdict
weighs the signal against the larger of the declared TOLERANCE and the noise
floor, at two stages, and splits into three outcomes.  The earlier row of five
steps said "one thing after another", which is not how the method works.

How it is drawn, and why (the previous draft was a literal translation):
  * Names and symbols only.  The sketch's three questions -- what the data
    say, what the data can resolve, how large a departure matters -- become
    one-word headings, Signal, Noise and Tolerance; the questions go to the
    caption.  No arrows inside boxes: a box names a step and the quantities
    it hands on.
  * Final size, sans-serif, as the Nature figure guide asks: Arial for text
    and Latin variables, Greek letters upright in Times New Roman standing in
    for the guide's Symbol font (see the rcParams), all text 7 pt, the
    guide's maximum.  Not smaller, because mathtext sets subscripts
    (true, 95, min, noise) at 70 % of the line: 4.9 pt at 7 pt, and further
    below the 5 pt minimum at any smaller size.  Lines and arrowheads are
    thinned to go with the type.
  * Colour marks where information comes from, nothing else: lavender for
    the signal, pale terracotta for the noise, neutral grey for the
    measurement and the declared tolerance.  The verdict is not a source, so
    it is white with a darker edge.  The outcomes keep teal / amber / grey.
  * Details go to the caption: stage two searches the collapse residual near
    alpha_bar; raw readings are kept; the resampled noise is the
    measurement's own residuals; sigma_true is sigma_alpha with sigma_noise
    removed (panel b).
  * The two signal boxes sit closer to each other than to the noise box, so
    the headings group them without an extra frame.
  * Right-angle connectors: one trunk leaves the measurement and forks just
    before the middle column, the three sources merge just after it, and one
    line enters the verdict.  Six slanted lines at six angles looked like
    clip art, and right angles match the bus under the verdict.
    CONNECT=fan brings the slanted version back.

A trial for the whole figure: panels a and b are still serif, and one figure
cannot mix the two, so they follow if this one reads well.

Layout as in fig1c_flow.py: one data unit is one inch, every box is sized to
its measured contents, and the printed clearances must stay positive.
Outputs: fig1c_flow_v2.png / .svg
"""

import os
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent

mpl.rcParams.update({
    # the family by NAME: with the generic "sans-serif", matplotlib 3.5 writes
    # 'sans-serif' into the SVG and the viewer picks whatever sans it has
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Nature's figure guide: text in Helvetica or Arial, Greek letters in the
    # Symbol font.  matplotlib cannot read Symbol (its legacy encoding leaves
    # every glyph missing, tested), so Greek letters are set upright in Times
    # New Roman, whose serif upright Greek is close to Symbol's; they reach it
    # through \mathsf, see upright_greek().  Latin variables are Arial
    # italic; digits, brackets, max and min Arial upright.  Earlier drafts
    # used Arial italic Greek (its tau is all but a capital T at 7 pt) and
    # then DejaVu Sans (clear, but neither Symbol nor a standard font).
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.fallback": "stixsans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 400,
})

C_TEXT = "#1f2937"
C_MUTED = "#5d6675"
C_ARROW = "#7b8794"
F_NEUT, E_NEUT = "#f2f4f6", "#c4cbd2"                   # measurement, tolerance
F_SIG, E_SIG, T_SIG = "#eceaf4", "#b4add4", "#3a3670"   # signal
F_NOI, E_NOI, T_NOI = "#f5e1d8", "#d6a18c", "#9a4527"   # noise
F_VER, E_VER = "#ffffff", "#5d6675"                     # verdict: not a source
PILLS = (("REDUCIBLE",  "#dceeed", "#86c2bb", "#1d6660"),
         ("NOT-RED",    "#f6ecd5", "#d8b56b", "#85580f"),
         ("UNRESOLVED", "#eceef0", "#c4cbd2", "#5d6675"))
# Stage two compares rho_true = sqrt(rho^2 - tau_min^2), not rho: v2(rt, tmin)
# in framework_pipeline.py, the mirror of stage one's sigma_true.  Written with a
# hat like sigma_true, since both are estimates (2026-09-15; the Verdict box and
# these two conditions said rho before).
CONDS = {
    # the two statistics take <= (2026-09-17, the author): the tolerance is declared as what
    # may be tolerated, so a departure exactly at it is still reducible.  The gate keeps its
    # strict <, which is what makes the UNRESOLVED line below exactly true
    "REDUCIBLE":  (r"$\hat\sigma_{\rm true}\leq\tau_1$ and $\hat\rho_{\rm true}\leq\tau$",
                   r"$q_{95}<\tau_1$ and $\tau_{\min}<\tau$"),
    "NOT-RED":    (r"$\hat\sigma_{\rm true}>\max(\tau_1,q_{95})$",
                   r"or $\hat\rho_{\rm true}>\max(\tau,\tau_{\min})$"),
    "UNRESOLVED": ("otherwise",
                   r"only when $q_{95}\geq\tau_1$ or $\tau_{\min}\geq\tau$"),
}

FS = 7.0                              # all text: the guide's maximum
LW_BOX, LW_VER, LW_ARROW = 0.6, 0.8, 0.75
HEAD = 6.5                            # arrowhead size (mutation scale)

#           name, what it hands on
MEASURE = ("Evaluations", [r"$M\times(K{+}1)\times n$"])   # "Measurement" until 2026-09-17
# lists put their commas OUTSIDE the math: inside, the comma plus a math space
# spread the symbols apart once the face was Arial
# alpha_bar IS the exponent pair (a, b) of the single group, so the box says so:
# without it, the collapse box's Pi_1^a Pi_2^b had exponents from nowhere
DIRECT = ("Directions", [r"$\bar\alpha\rightarrow(a,b)$, $\sigma_\alpha$"])
# "Collapse" alone did not say what collapses or onto what.  The response Pi_o
# (the same Pi_o as panel a) is fitted as ONE function g of the single group
# Pi_1^a Pi_2^b, whose exponents are the direction alpha_bar from stage one;
# rho is what is left over.  "Residual" is the brief's own word (collapse
# residual).  The search window around alpha_bar and the polynomial degree
# belong in the caption.
COLLAPSE = ("Collapse", [r"$\Pi_o\approx g(\Pi_1^{a}\Pi_2^{b})$",
                         r"Residual $\rho$"])
# eps_hat is the pool's own scale, the estimate the other three are built from
# (tau_min = eps_hat / (sqrt(n) s_y), and the null draws its noise from the pool)
# one line, not two: a second line brought the box's bottom to within 0.6 mm of the
# branch line that runs under it
NOISE = ("Residual pool", [r"$\hat\varepsilon$, $\sigma_{\rm noise}$, $q_{95}$, $\tau_{\min}$"])
VERDICT = ("Verdict",
           [r"Stage one: $\hat\sigma_{\rm true}$ vs $\max(\tau_1,q_{95})$",
            r"Stage two: $\hat\rho_{\rm true}$ vs $\max(\tau,\tau_{\min})$"])
TOLER = r"$\tau_1$, $\tau$"

GREEK = r"(alpha|sigma|tau|rho|varepsilon|Pi)"


def upright_greek(s):
    """Route every Greek letter to mathtext.sf (Times New Roman, upright).

    The strings above stay readable as ordinary TeX; this rewrites them once,
    before anything is measured or drawn.  An accented letter keeps its
    accent outside (hat{sf{sigma}}); a bare one gets braces round the font
    switch so a sub- or superscript can still attach to it.
    """
    s = re.sub(r"\\(hat|bar)\\" + GREEK + r"(?![A-Za-z])",
               r"\\\1{\\mathsf{\\\2}}", s)
    return re.sub(r"(?<!\\mathsf\{)\\" + GREEK + r"(?![A-Za-z])",
                  r"{\\mathsf{\\\1}}", s)


MEASURE, DIRECT, COLLAPSE, NOISE, VERDICT = [
    (name, [upright_greek(s) for s in lines])
    for name, lines in (MEASURE, DIRECT, COLLAPSE, NOISE, VERDICT)]
TOLER = upright_greek(TOLER)
CONDS = {k: tuple(upright_greek(s) for s in v) for k, v in CONDS.items()}

# ---- measure the type before choosing any width ---------------------------
PROBE_DPI = 200.0
_probe = plt.figure(figsize=(6, 2), dpi=PROBE_DPI)


def width_in(s, weight="normal"):
    """Width of a rendered 7 pt string, in inches."""
    t = _probe.text(0.0, 0.0, s, fontsize=FS, weight=weight)
    _probe.canvas.draw()
    w = t.get_window_extent(_probe.canvas.get_renderer()).width / PROBE_DPI
    t.remove()
    return w


def need(name, lines):
    return max(width_in(s) for s in [name] + lines)


PAD_BOX = 0.08                        # clear space inside a box, each side
PAD_T, PAD_B = 0.10, 0.09             # above the name, below the last line
G_TL, G_LL = 0.135, 0.125             # name to first line, line to line
H2 = PAD_T + G_TL + PAD_B             # name + one line
H3 = H2 + G_LL                        # name + two lines
H_TOL = 0.22                          # the tolerance: one line, no name
ROUND = 0.05

# the measurement and the three sources share one width, so the two columns
# line up
W_SRC = max(need(*b) for b in (MEASURE, DIRECT, COLLAPSE, NOISE)) + 2 * PAD_BOX
W_VER = need(*VERDICT) + 2 * PAD_BOX
W_TOL = width_in(TOLER) + 2 * 0.10
need_pill = max([width_in(n) for n, _, _, _ in PILLS] +
                [width_in(l) for pair in CONDS.values() for l in pair])
PW, PH = need_pill + 2 * PAD_BOX, H3

# ---- geometry ----------------------------------------------------------------
G12_MIN, G23_MIN, G_OUT_MIN = 0.45, 0.55, 0.30
top_min = 2 * W_SRC + W_VER + G12_MIN + G23_MIN
# UNRESOLVED hangs straight under the verdict, REDUCIBLE starts at the left
# edge and NOT-RED sits halfway; this much width keeps their gaps open
row_min = W_VER / 2 + 2.5 * PW + 2 * G_OUT_MIN
PANEL_W = max(top_min, row_min)
extra = PANEL_W - top_min             # shared out between the slanted gaps
G12 = G12_MIN + extra * G12_MIN / (G12_MIN + G23_MIN)
G23 = G23_MIN + extra * G23_MIN / (G12_MIN + G23_MIN)
X2 = W_SRC + G12
X3 = X2 + W_SRC + G23
XC2, XV = X2 + W_SRC / 2, X3 + W_VER / 2

GAP_SIG = 0.12                        # between the two signal boxes: one group
GAP_NOI = 0.25                        # before the noise box: room for "Noise"
HEAD_GAP = 0.05                       # a heading's bottom above its box
# the group's stacked scripts (a over 1, b over 2) need room above and below;
# with Times New Roman Pi they stand taller than they did in Arial
G_TL_C, G_LL_C = G_TL + 0.025, G_LL + 0.05
H_C = PAD_T + G_TL_C + (len(COLLAPSE[1]) - 1) * G_LL_C + PAD_B
C_TOP, C_BOT = H_C / 2, -H_C / 2      # y = 0: collapse, measurement, verdict
D_BOT = C_TOP + GAP_SIG
D_TOP = D_BOT + H2
N_TOP = C_BOT - GAP_NOI
N_BOT = N_TOP - H2
D_MID, N_MID = (D_TOP + D_BOT) / 2, (N_TOP + N_BOT) / 2
M_TOP = H2 / 2
V_TOP, V_BOT = H3 / 2, -H3 / 2
T_TOP = D_TOP                         # the tolerance level with the top box
T_BOT = T_TOP - H_TOL

BUS_Y = min(N_BOT, V_BOT) - 0.16
Y_PILL = BUS_Y - 0.12 - PH
pcx = [PW / 2, 0.0, XV]
pcx[1] = 0.5 * (pcx[0] + pcx[2])

MARGIN = 0.05
X0F = -MARGIN
X1F = max(PANEL_W, XV + PW / 2) + MARGIN
Y1F = D_TOP + HEAD_GAP + 0.12 + MARGIN
Y0F = Y_PILL - MARGIN

# one data unit = one inch: axes fills the figure, figure matches the span
fig = plt.figure(figsize=(X1F - X0F, Y1F - Y0F), dpi=220)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(X0F, X1F)
ax.set_ylim(Y0F, Y1F)
ax.axis("off")

FITS = []                             # (what, artist, allowed x0, x1)


def box(x, top, w, name, lines, fc, ec, tc, lw=LW_BOX, pitch=G_LL,
        g_tl=G_TL):
    h = PAD_T + g_tl + (len(lines) - 1) * pitch + PAD_B
    ax.add_patch(FancyBboxPatch((x, top - h), w, h,
                                boxstyle="round,pad=0.0,rounding_size=%.3f"
                                % ROUND, facecolor=fc, edgecolor=ec, lw=lw,
                                zorder=3))
    y = top - PAD_T
    a = ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=FS,
                color=tc, zorder=4)
    FITS.append((name, a, x, x + w))
    for i, s in enumerate(lines):
        b = ax.text(x + w / 2, y - g_tl - i * pitch, s, ha="center",
                    va="center", fontsize=FS, color=C_MUTED, zorder=4)
        FITS.append((s, b, x, x + w))


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=HEAD, lw=LW_ARROW,
                                 color=C_ARROW, zorder=2,
                                 shrinkA=0, shrinkB=0))


def heading(x, y, s, colour):
    return ax.text(x, y, s, ha="center", va="bottom", fontsize=FS,
                   weight="bold", color=colour, zorder=4)


box(0.0, M_TOP, W_SRC, *MEASURE, F_NEUT, E_NEUT, C_TEXT)
box(X2, D_TOP, W_SRC, *DIRECT, F_SIG, E_SIG, T_SIG)
box(X2, C_TOP, W_SRC, *COLLAPSE, F_SIG, E_SIG, T_SIG, pitch=G_LL_C,
    g_tl=G_TL_C)
box(X2, N_TOP, W_SRC, *NOISE, F_NOI, E_NOI, T_NOI)
box(X3, V_TOP, W_VER, *VERDICT, F_VER, E_VER, C_TEXT, lw=LW_VER)
ax.add_patch(FancyBboxPatch((XV - W_TOL / 2, T_BOT), W_TOL, H_TOL,
                            boxstyle="round,pad=0.0,rounding_size=%.3f"
                            % ROUND, facecolor=F_NEUT, edgecolor=E_NEUT,
                            lw=LW_BOX, zorder=3))
t_tol = ax.text(XV, T_BOT + H_TOL / 2, TOLER, ha="center", va="center",
                fontsize=FS, color=C_MUTED, zorder=4)
FITS.append(("tolerance", t_tol, XV - W_TOL / 2, XV + W_TOL / 2))

h_sig = heading(XC2, D_TOP + HEAD_GAP, "Signal", T_SIG)
h_noi = heading(XC2, N_TOP + HEAD_GAP, "Noise", T_NOI)
h_tol = heading(XV, T_TOP + HEAD_GAP, "Tolerance", C_MUTED)

CONNECT = os.environ.get("CONNECT", "bracket")      # bracket | fan
HUG = 0.18                            # fork / merge line to the middle column


def line(xs, ys):
    ax.plot(xs, ys, color=C_ARROW, lw=LW_ARROW, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=2)


if CONNECT == "bracket":
    # One trunk leaves the measurement and forks just before the middle
    # column; the three sources merge just after it; one line enters the
    # verdict.  The fork and the merge are drawn as single polylines so
    # their corners join cleanly; the arrows only add the heads.
    x_fork, x_merge = X2 - HUG, X2 + W_SRC + HUG
    line([W_SRC, x_fork], [0.0, 0.0])
    line([X2, x_fork, x_fork, X2], [D_MID, D_MID, N_MID, N_MID])
    line([X2 + W_SRC, x_merge, x_merge, X2 + W_SRC],
         [D_MID, D_MID, N_MID, N_MID])
    line([X2 + W_SRC, x_merge], [0.0, 0.0])
    for y in (D_MID, 0.0, N_MID):
        arrow((x_fork, y), (X2, y))
    arrow((x_merge, 0.0), (X3, 0.0))
else:
    # the slanted version: six lines, three heads into the verdict
    OFF_M, OFF_V = 0.07, 0.12         # where arrows leave / enter a box edge
    arrow((W_SRC, OFF_M), (X2, D_MID))
    arrow((W_SRC, 0.0), (X2, 0.0))
    arrow((W_SRC, -OFF_M), (X2, N_MID))
    arrow((X2 + W_SRC, D_MID), (X3, OFF_V))
    arrow((X2 + W_SRC, 0.0), (X3, 0.0))
    arrow((X2 + W_SRC, N_MID), (X3, -OFF_V))
arrow((XV, T_BOT), (XV, V_TOP))

# ---- the three outcomes -------------------------------------------------------
ax.plot([XV, XV, pcx[0]], [V_BOT, BUS_Y, BUS_Y], color=C_ARROW, lw=LW_ARROW,
        solid_joinstyle="round", zorder=2)
for (name, fc, ec, tc), cx in zip(PILLS, pcx):
    ax.add_patch(FancyBboxPatch((cx - PW / 2, Y_PILL), PW, PH,
                                boxstyle="round,pad=0.0,rounding_size=%.3f"
                                % ROUND, facecolor=fc, edgecolor=ec,
                                lw=LW_BOX, zorder=3))
    y_name = Y_PILL + PH - PAD_T
    a = ax.text(cx, y_name, name, ha="center", va="center", fontsize=FS,
                color=tc, zorder=4)
    FITS.append((name, a, cx - PW / 2, cx + PW / 2))
    for i, l in enumerate(CONDS[name]):
        b = ax.text(cx, y_name - G_TL - i * G_LL, l, ha="center",
                    va="center", fontsize=FS, color=C_MUTED, zorder=4)
        FITS.append((name + " condition", b, cx - PW / 2, cx + PW / 2))
    arrow((cx, BUS_Y), (cx, Y_PILL + PH))

# ---- checks -----------------------------------------------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()


def bb(art):
    return art.get_window_extent(rend).transformed(ax.transData.inverted())


worst = min(((min(bb(a).x0 - x0, x1 - bb(a).x1), what)
             for what, a, x0, x1 in FITS), key=lambda t: t[0])
gaps_out = [pcx[i + 1] - pcx[i] - PW for i in range(2)]
# a list, not a bare string: a bare string is parsed as a fontconfig pattern,
# and the hyphen in "sans-serif" breaks that parser
face = Path(font_manager.findfont(
    font_manager.FontProperties(family=mpl.rcParams["font.family"]))).name
print("text face %s; all text %.1f pt, subscripts %.1f pt" % (face, FS, 0.7 * FS))
print("headings: Signal %.3f in above its box; Noise %.3f in above its box "
      "and %.3f in below the collapse box; Tolerance %.3f in above its box"
      % (bb(h_sig).y0 - D_TOP, bb(h_noi).y0 - N_TOP, C_BOT - bb(h_noi).y1,
         bb(h_tol).y0 - T_TOP))

for ext in ("png", "svg"):
    fig.savefig(ROOT / ("fig1c_flow_v2.%s" % ext), bbox_inches="tight",
                transparent=True, pad_inches=0.02)
plt.close(fig)
plt.close(_probe)

W, H = X1F - X0F, Y1F - Y0F
print("widths: sources %.2f, verdict %.2f, tolerance %.2f, outcomes %.2f in;"
      "  slanted gaps %.2f / %.2f in" % (W_SRC, W_VER, W_TOL, PW, G12, G23))
print("figure %.2f x %.2f in = %.0f x %.0f mm at final size"
      % (W, H, W * 25.4, H * 25.4))
print("tightest label clearance inside a box %+.3f in  (%s)" % worst)
print("gaps between outcome boxes %s in" % ["%.2f" % g for g in gaps_out])
print("Saved fig1c_flow_v2.png / .svg")
