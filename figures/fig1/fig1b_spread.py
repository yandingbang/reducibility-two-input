# -*- coding: utf-8 -*-
"""Figure 1b -- the M directions, and what their spread splits into.

Panel a shows the directions in place.  This panel does the one thing that
makes the method work: the observed spread is NOT the answer, because it
carries the noise as well.

Two parts, each with one job:

  the fan        M rays on the axial half-circle, the thick one their mean.
                 Qualitative: the readings nearly agree, and the fan is narrow
                 against the whole range of distinct directions.  An arc with
                 end ticks marks mean -/+ sigma_alpha and is labelled
                 +-sigma_alpha: its whole width is 2 sigma_alpha, so a bare
                 sigma_alpha label would invite reading the arc length as
                 sigma_alpha.  It is not the range of the readings either;
                 the range grows with M, the standard deviation does not.

  the triangle   quantitative, and this is where a careless panel goes wrong.
                 The decomposition is a sum of SQUARES,
                     sigma_alpha^2 = sigma_hat_true^2 + sigma_noise^2,
                 so sigma_noise must NOT be drawn as a sub-interval of
                 sigma_alpha -- that would read as linear subtraction.  A sum
                 of squares is Pythagoras, so: legs sigma_hat_true and
                 sigma_noise, hypotenuse sigma_alpha.  It sits in the empty
                 upper-left of the half-circle.  The leg carries the hat, as
                 the text does (2026-09-17): what the triangle solves for is
                 the estimate, not the field's own sigma_true.

The triangle's PROPORTIONS are real, from the low-Re benchmark (sigma_alpha
1.077, sigma_noise 0.921, sigma_hat_true 0.559 deg; the identity closes to
three decimals).  Only the fan's angular width is exaggerated, for the same reason
as panel a: the real sigma_alpha is about 1 deg.

Colour carries over from panel a: indigo = what was measured, terracotta =
the noise, and sigma_hat_true -- the spread left when the noise is taken out,
the quantity stage one judges -- is a deeper step of the field's own Blues.  (It was
near-black, which could not be told from the indigo hypotenuse.)

Style: the rays start at 0.30 R and are light, so they no longer pile into a
dark hub at the centre that swallowed the mean; a white line under the mean's
SHAFT only (not its head, not the centre) separates it from them.  The
half-circle is a thin solid line like the frames of panel a, and the
triangle's edges meet with round ends instead of notched butt ends.

Title: "Separating noise from the spread".  It was "The spread is not the
answer", a negation that needed the caption and never named the noise.

Type size (2026-09-15): every text is 7 pt, the maximum Nature Communications
allows at final size, as in fig1c_flow_v2.py; subscripts come out at 4.9 pt.
The geometry keeps its physical size, so the saved crop IS the final size:
place the PNG / SVG in Figure 1 at 100 %, without scaling.  The text was 9 to
12.5 pt before, i.e. too large at this size.
"""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

ROOT = Path(__file__).parent

# Nature's figure guide: text in Helvetica or Arial, Greek letters in the
# Symbol font.  matplotlib cannot read Symbol (every glyph comes out missing),
# so Greek letters are set upright in Times New Roman, whose serif upright
# Greek is close to Symbol's.  Latin variables are Arial italic; digits,
# brackets and operators Arial upright.  Same setup as fig1c_flow_v2.py.
mpl.rcParams.update({
    "font.family": "Arial",          # by name: 'sans-serif' went into the SVG
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

GREEK = r"(alpha|sigma|tau|rho|varepsilon|pi|Pi)"


def upright_greek(s):
    """Route every Greek letter of a math string to mathtext.sf, upright."""
    s = re.sub(r"\\(hat|bar)\\" + GREEK + r"(?![A-Za-z])",
               r"\\\1{\\mathsf{\\\2}}", s)
    return re.sub(r"(?<!\\mathsf\{)\\" + GREEK + r"(?![A-Za-z])",
                  r"{\\mathsf{\\\1}}", s)


# Applied to EVERY text as it is set, annotations included, so no label can
# slip through with an italic Arial Greek letter.
_set_text = Text.set_text
Text.set_text = lambda self, s: _set_text(
    self, upright_greek(s) if isinstance(s, str) and "$" in s else s)

C_MEAS = "#33305f"
C_NOISE = "#b54f30"
C_TRUE = "#2171b5"       # deeper step of panel a's Blues: the TRUE field
C_TEXT = "#1f2937"
C_MUTED = "#5d6675"
C_SOFT = "#c8d0d8"
FS = 7.0                 # every text: Nature Communications' maximum at final size

# ---- knobs -----------------------------------------------------------------
M = int(os.environ.get("M", "40"))
FAN_DEG = float(os.environ.get("FAN_DEG", "13.0"))   # half-width, EXAGGERATED
ABAR = float(os.environ.get("ABAR", "62.0"))
SEED = int(os.environ.get("SEED", "11"))
FAN_WEIGHT = os.environ.get("FAN_WEIGHT", "1") == "1"   # rays fade in the tails
# The "Observed spread" phrase is off by default: +-sigma_alpha on the arc over
# the measured rays, and the same indigo sigma_alpha on the hypotenuse, already
# say it, and the phrase was the widest thing in the panel.  The caption
# defines sigma_alpha as the observed spread.  SHOW_OBS=1 puts it back.
SHOW_OBS = os.environ.get("SHOW_OBS", "0") == "1"
# 公式与那两行说明默认不画:三角形本身就是那条公式(两条直角边、斜边、直角
# 记号), 再写一遍代数式是同一句话说两次;而「sigma_noise 由测量自带的重复
# 算出」是题注该承担的。SHOW_EQ=1 可以放回来。
SHOW_EQ = os.environ.get("SHOW_EQ", "0") == "1"

SA, SN, ST = 1.077, 0.921, 0.559        # real, low-Re benchmark (deg)

R = 1.0
fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=220)
ax.set_aspect("equal"); ax.axis("off")

# ---- the axial half-circle -------------------------------------------------
th = np.radians(np.linspace(0, 180, 400))
ax.plot(R * np.cos(th), R * np.sin(th), color=C_SOFT, lw=1.0, zorder=1)
ax.plot([-R, R], [0, 0], color=C_SOFT, lw=0.9, zorder=1)
# 0 and pi label the two ENDS OF THE DIAMETER, so they sit on its line, just
# beyond each end.  They used to sit above the line (bottom at +0.045), which
# made them read as labels of the arc rather than of the endpoints.
END_GAP = 0.05
t_zero = ax.text(R + END_GAP, 0.0, r"$0$", ha="left", va="center",
                 fontsize=FS, color=C_MUTED)
t_pi = ax.text(-R - END_GAP, 0.0, r"$\pi$", ha="right", va="center",
               fontsize=FS, color=C_MUTED)
ax.text(0.34, -0.11, r"A direction is defined mod $\pi$", ha="center",
        va="top", fontsize=FS, color=C_MUTED, style="italic")

# ---- the fan ---------------------------------------------------------------
rng = np.random.default_rng(SEED)
ang = ABAR + rng.normal(0.0, FAN_DEG / 2.0, M)
# Each ray's opacity follows the normal density at its angle, so the fan's
# visible body is its core rather than its two most extreme rays.  At one
# opacity the fan's edge was set by rays near +-2 sd, while the bracket marks
# +-1 sd (which is what sigma_alpha is), so the bracket looked too short for
# the fan.  Widening the bracket instead would label a range as sigma_alpha.
# FAN_WEIGHT=0 restores rays of equal opacity.
_z = (ang - ang.mean()) / np.std(ang, ddof=1)
_w = np.exp(-0.5 * _z ** 2) if FAN_WEIGHT else np.full_like(_z, 0.55)
for a, w in zip(ang, _w):
    aa = np.radians(a)
    # start away from the centre: from 0 they piled into a dark hub
    ax.plot([R * 0.30 * np.cos(aa), R * 0.92 * np.cos(aa)],
            [R * 0.30 * np.sin(aa), R * 0.92 * np.sin(aa)],
            color=C_MEAS, lw=0.55, alpha=0.08 + 0.55 * w, zorder=3,
            solid_capstyle="round")
am = np.radians(ABAR)
# a plain white line under the mean's SHAFT, only where the rays are: it lifts
# the mean off them without cutting a notch into the diameter at the centre
# (a halo from the centre did) or leaving a white blob round the head (a halo
# on the whole arrow did)
ax.plot([R * 0.28 * np.cos(am), R * 0.86 * np.cos(am)],
        [R * 0.28 * np.sin(am), R * 0.86 * np.sin(am)],
        color="white", lw=4.2, solid_capstyle="round", zorder=4.9)
ax.add_patch(FancyArrowPatch((0, 0),
                             (R * 0.99 * np.cos(am), R * 0.99 * np.sin(am)),
                             arrowstyle="-|>", mutation_scale=12, lw=1.9,
                             color=C_MEAS, zorder=5, shrinkA=0, shrinkB=0))
# the mean's label goes on the COUNTER-clockwise side, mirroring the spread
# label on the other -- put both on the same side and they overlap the bracket
AR = np.radians(ABAR + 14.0)
ax.plot([R * 1.01 * np.cos(am), R * 1.12 * np.cos(AR)],
        [R * 1.01 * np.sin(am), R * 1.12 * np.sin(AR)],
        color=C_MEAS, lw=0.7, alpha=0.7, zorder=5)
ax.text(R * 1.16 * np.cos(AR), R * 1.16 * np.sin(AR), r"$\bar\alpha$",
        ha="right", va="center", fontsize=FS, color=C_MEAS)

# bracket for the observed spread, label placed on the CLOCKWISE side so it
# cannot collide with the mean's own label
sd = float(np.std(ang, ddof=1))
br = np.radians(np.linspace(ABAR - sd, ABAR + sd, 60))
ax.plot(R * 1.12 * np.cos(br), R * 1.12 * np.sin(br), color=C_MEAS, lw=1.1,
        zorder=5)
for sgn in (-1, 1):
    aa = np.radians(ABAR + sgn * sd)
    ax.plot([R * 1.08 * np.cos(aa), R * 1.16 * np.cos(aa)],
            [R * 1.08 * np.sin(aa), R * 1.16 * np.sin(aa)],
            color=C_MEAS, lw=1.0, zorder=5)
# The label hangs just right of the bracket, vertically centred on the
# bracket's middle.  It used to sit at 1.22 R and 2.4 sd clockwise of the mean,
# which put it well below and to the right of the bracket, detached from it.
# Both numbers come from the bracket as drawn, so the label follows it if the
# fan width or the mean direction change.
_bt = np.radians(ABAR + np.array([-1.0, 1.0]) * sd)
BR_RIGHT = float(max((R * 1.12 * np.cos(br)).max(),
                     (R * 1.16 * np.cos(_bt)).max()))
BR_MIDY = float(R * 1.12 * np.sin(am))
# +-sigma_alpha, not sigma_alpha: the arc runs from mean - sigma_alpha to
# mean + sigma_alpha, so its whole width is 2 sigma_alpha
t_sa = ax.text(BR_RIGHT + 0.06, BR_MIDY, r"$\pm\sigma_\alpha$", ha="left",
               va="center", fontsize=FS, color=C_MEAS)
# on the SAME line as the symbol, to its right, vertically centred with it.
# Placed a fixed 7 pt after the symbol AS DRAWN, so the wider +- symbol cannot
# run into it (a fixed data offset of 0.23 was sized for sigma_alpha alone).
if SHOW_OBS:
    ax.annotate("Observed spread", xy=(1.0, 0.5), xycoords=t_sa,
                xytext=(7, 0), textcoords="offset points", ha="left",
                va="center", fontsize=FS, color=C_MEAS)

# ---- Pythagoras, OUTSIDE the half-circle -----------------------------------
# Below the baseline and left of centre: clear space, and the panel then reads
# top-to-bottom as "what was observed" then "what it splits into".  The indigo
# hypotenuse is the same colour as the fan, which is what ties the two halves
# together -- no leader line needed.
SC = 0.50 / SN
a_leg, b_leg = ST * SC, SN * SC
# nudged 0.06 right and 0.06 down from (-1.30, 0.86); the labels move with it,
# and the check before saving reports how close they come to the arc
tx = float(os.environ.get("TX", "-1.24"))
ty = float(os.environ.get("TY", "0.80"))

ax.plot([tx, tx + a_leg], [ty, ty], color=C_TRUE, lw=2.2, zorder=6,
        solid_capstyle="round")
ax.plot([tx + a_leg, tx + a_leg], [ty, ty + b_leg], color=C_NOISE, lw=2.2,
        zorder=6, solid_capstyle="round")
ax.plot([tx, tx + a_leg], [ty, ty + b_leg], color=C_MEAS, lw=2.2, zorder=6,
        solid_capstyle="round")
q = 0.05
ax.plot([tx + a_leg - q, tx + a_leg - q, tx + a_leg], [ty, ty + q, ty + q],
        color=C_MUTED, lw=0.8, zorder=6)

t_true = ax.text(tx + a_leg / 2, ty - 0.045, r"$\hat\sigma_{\rm true}$",
                 ha="center", va="top", fontsize=FS, color=C_TRUE)
t_noise = ax.text(tx + a_leg + 0.045, ty + b_leg / 2, r"$\sigma_{\rm noise}$",
                  ha="left", va="center", fontsize=FS, color=C_NOISE)
# sigma_alpha sits off the MIDDLE of the hypotenuse, pushed out along its
# normal.  It used to be placed left of the triangle's left corner, and at
# that height the hypotenuse has already moved well to the right, so the label
# floated far from its own edge.  The offset is chosen so the label's nearest
# corner ends up about as far from the hypotenuse as the other two labels are
# from their legs; the check before saving prints all three.
HYP_N = np.array([-b_leg, a_leg]) / np.hypot(a_leg, b_leg)   # outward normal
hyp_mid = np.array([tx + a_leg / 2, ty + b_leg / 2])
# 0.072: at 7 pt the offset of 0.091 (tuned for 11.5 pt) gave a gap of 0.064
# against 0.045 for the other two, and the gap moves one-for-one with this offset
lab_a = hyp_mid + 0.072 * HYP_N
t_alpha = ax.text(lab_a[0], lab_a[1], r"$\sigma_\alpha$", ha="right",
                  va="center", fontsize=FS, color=C_MEAS)
# "a sum of squares, not an interval to cut" was here.  It collided with the
# pi label and the mod-pi note, and it is a caption remark, not a figure
# label -- the triangle already says it.  Moved to the caption.

# ---- title, identity, and the sentence that matters ------------------------
# 1.40 R, down from 1.50 R: the title floated well above the triangle's top
# corner, the highest thing under it.  The check before saving prints the gap.
TITLE_Y = 1.40
TITLE_FS = FS            # 11.5 originally, then 10.5 and 9.5; 7 pt since 2026-09-15
t_title = ax.text(0, R * TITLE_Y, "Separating noise from the spread",
                  ha="center", va="bottom", fontsize=TITLE_FS, weight="bold",
                  color=C_TEXT)
if SHOW_EQ:
    ax.text(0, -0.46,
            r"$\sigma_\alpha^2=\sigma_{\rm true}^2+\sigma_{\rm noise}^2$",
            ha="center", va="top", fontsize=FS, color=C_TEXT)
    ax.text(0, -0.68,
            "$\\sigma_{\\rm noise}$ is computed from the measurement's own\n"
            "repeats, so what gets judged is $\\sigma_{\\rm true}$",
            ha="center", va="top", fontsize=FS, color=C_MUTED,
            linespacing=1.45)

ax.set_xlim(min(-1.28, tx - 0.26), 2.02)
ax.set_ylim(min(ty - 0.22, -0.94 if SHOW_EQ else -0.34),
            max(R * 1.60, ty + b_leg + 0.22))

# how far each triangle label sits from its own edge, in data units
fig.canvas.draw()
_rend = fig.canvas.get_renderer()


def _box(t):
    return t.get_window_extent(_rend).transformed(ax.transData.inverted())


_bb = _box(t_alpha)
_corners = np.array([[_bb.x0, _bb.y0], [_bb.x1, _bb.y0],
                     [_bb.x0, _bb.y1], [_bb.x1, _bb.y1]])
gap_alpha = float(((_corners - np.array([tx, ty])) @ HYP_N).min())
gap_noise = float(_box(t_noise).x0 - (tx + a_leg))
gap_true = float(ty - _box(t_true).y1)
print("label gaps to their own edges: sigma_alpha %.3f   sigma_noise %.3f   "
      "sigma_true %.3f" % (gap_alpha, gap_noise, gap_true))
_bz, _bp = _box(t_zero), _box(t_pi)
print("0 and pi on the diameter: centres at y %+.3f and %+.3f; gaps to the "
      "ends %.3f and %.3f" % (0.5 * (_bz.y0 + _bz.y1), 0.5 * (_bp.y0 + _bp.y1),
                              _bz.x0 - R, -R - _bp.x1))
_bs = _box(t_sa)
print("sigma_alpha label: %.3f right of the bracket, centre %+.3f from the "
      "bracket's middle height" % (_bs.x0 - BR_RIGHT,
                                   0.5 * (_bs.y0 + _bs.y1) - BR_MIDY))
# nearest approach of each triangle label to the half-circle arc (upper half)
_arc_clear = []
for _t in (t_noise, t_true, t_alpha):
    _b = _box(_t)
    _gx, _gy = np.meshgrid(np.linspace(_b.x0, _b.x1, 25),
                           np.linspace(_b.y0, _b.y1, 9))
    _up = _gy > 0
    _arc_clear.append(float(np.abs(np.hypot(_gx[_up], _gy[_up]) - R).min())
                      if _up.any() else float("inf"))
print("triangle labels' clearance from the arc: sigma_noise %.3f   "
      "sigma_true %.3f   sigma_alpha %.3f" % tuple(_arc_clear))
_under = [a for a in list(ax.texts) + list(ax.lines) + list(ax.patches)
          if a is not t_title and a.get_visible()]
_top_under = max(a.get_window_extent(_rend).transformed(
    ax.transData.inverted()).y1 for a in _under)
print("title: bottom %.3f above the highest element under it"
      % (_box(t_title).y0 - _top_under))

# Crop to what is actually drawn.  "tight" also kept the axes' own extent, and
# xlim runs to 2.02 to leave room for the old "Observed spread" phrase, so with
# the phrase gone the panel kept a blank strip on the right.  Narrowing xlim
# instead would rescale the geometry against the fixed-size text.
from matplotlib.transforms import Bbox
_drawn = [a.get_window_extent(_rend)
          for a in list(ax.texts) + list(ax.lines) + list(ax.patches)
          if a.get_visible()]
CROP = Bbox.union(_drawn).transformed(fig.dpi_scale_trans.inverted())
CROP = CROP.padded(0.04)
print("saved crop %.2f x %.2f in = %.1f x %.1f mm: the final size, place at 100 %%"
      % (CROP.width, CROP.height, CROP.width * 25.4, CROP.height * 25.4))
for ext in ("png", "svg"):
    fig.savefig(ROOT / ("fig1b_spread.%s" % ext), bbox_inches=CROP,
                transparent=True)
plt.close(fig)

print("M %d   fan half-width %.1f deg (exaggerated)   drawn sd %.2f deg"
      % (M, FAN_DEG, sd))
print("rays reach %+.1f to %+.1f sd from the mean; the bracket marks -1 to +1 "
      "sd; tail rays %s" % (_z.min(), _z.max(),
                            "fade with the normal density" if FAN_WEIGHT
                            else "at one opacity"))
print("triangle from the low-Re benchmark: legs %.3f / %.3f  ->  hypotenuse "
      "%.4f   vs  sigma_alpha %.3f" % (ST, SN, np.hypot(ST, SN), SA))
corners = [(tx, ty), (tx + a_leg, ty), (tx + a_leg, ty + b_leg)]
inside = [c for c in corners if np.hypot(*c) < R and c[1] > 0]
print("triangle bbox  x %.2f-%.2f  y %.2f-%.2f" % (tx, tx + a_leg, ty, ty + b_leg))
print("   entirely outside the half-circle: %s" % (not inside))
print("Saved fig1b_spread.png / .svg")
