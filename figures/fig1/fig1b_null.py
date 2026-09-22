# -*- coding: utf-8 -*-
"""Figure 1b, third sub-panel -- what noise alone produces, and where q95 is.

Why this panel exists.  q95 is used in panel c (stage one compares sigma_true
with max(tau_1, q95)) but nothing before it says what q95 is: the spread
sub-panel stops at sigma_noise.  Once panel d was merged into panel c, q95 had
no introduction at all.  This panel is that introduction: run the same pipeline
on data with the structure removed and the measured noise put back, collect the
noise-corrected spread sigma_true each time, and take its 95th percentile.

Schematic in style, real in shape.  No tick values except 0; but the histogram
is the actual null distribution of the low-Re rough-pipe benchmark, 6400
samples, which is also the benchmark whose spread the triangle decomposes.
Its shape is part of the message: after the noise is subtracted, most
noise-only runs give EXACTLY zero (about 61 % here, 59-65 % across the eight
benchmarks), so the distribution is a spike at 0 plus a right tail.  The spike
is far taller than the tail, so it is drawn clipped, with a break, and its
share written on it.

Colours follow Figure 1: terracotta = produced by the noise (the read-out
wedge, the sigma_noise leg, the q95 tick in the verdict geometry), blue =
sigma_true as in the spread sub-panel.  The top 5 % beyond q95 is the darker
terracotta, which is what "95th percentile" means drawn.

Caption: this is one campaign's null distribution; q95 moves between campaigns
(15.4 % spread for this benchmark over 24 independent campaigns).

Outputs: fig1b_null.png / .svg
"""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Bbox
from matplotlib.text import Text

ROOT = Path(__file__).parent
DATA = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "data" / "fig2_method_data.npz"
BENCH = os.environ.get("BENCH", "u0")        # u0 = rough pipe, low Re

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


# Applied to EVERY text as it is set, so no label can slip through with an
# italic Arial Greek letter.
_set_text = Text.set_text
Text.set_text = lambda self, s: _set_text(
    self, upright_greek(s) if isinstance(s, str) and "$" in s else s)

C_NOISE = "#b54f30"          # produced by the noise
C_NOISE_PALE = "#eed3c8"     # the 95 % below q95
C_NOISE_DARK = "#c9714f"     # the 5 % above it
C_TRUE = "#2171b5"           # sigma_true, as in the spread sub-panel
C_TEXT = "#1f2937"
C_MUTED = "#5d6675"
C_AXIS = "#7b8794"
TITLE_FS = 9.5               # matches the other sub-panel titles
NB = int(os.environ.get("NB", "16"))          # tail bins

# ---- the null distribution ---------------------------------------------------
d = np.load(DATA)
null = np.asarray(d[BENCH + "_st_null"], dtype=float)
q95 = float(d[BENCH + "_q95"])
name = str(d[BENCH + "_name"])
N = null.size
zero_frac = float(np.mean(null == 0.0))
tail = null[null > 0.0]

XMAX = float(np.percentile(null, 99.5)) * 1.15
edges = np.linspace(0.0, XMAX, NB + 1)
counts, _ = np.histogram(tail, bins=edges)
frac = counts / N                              # share of ALL samples per bin
bw = edges[1] - edges[0]
CAP = 2.4 * float(frac.max())                  # the zero spike is clipped here

fig, ax = plt.subplots(figsize=(2.9, 2.2), dpi=220)
ax.axis("off")

# ---- tail bars: pale below q95, darker above it -----------------------------
for i in range(NB):
    x0, x1, h = edges[i], edges[i + 1], frac[i]
    if h <= 0:
        continue
    pieces = []
    if x1 <= q95:
        pieces.append((x0, x1, C_NOISE_PALE))
    elif x0 >= q95:
        pieces.append((x0, x1, C_NOISE_DARK))
    else:
        pieces += [(x0, q95, C_NOISE_PALE), (q95, x1, C_NOISE_DARK)]
    for a, b, c in pieces:
        ax.add_patch(Rectangle((a, 0.0), b - a, h, facecolor=c,
                               edgecolor="none", zorder=2))
    # white separators between bars
    ax.plot([x1, x1], [0.0, max(h, frac[i + 1] if i + 1 < NB else 0.0)],
            color="white", lw=0.6, zorder=3)

# ---- the spike at exactly zero, clipped with a break ------------------------
SW = 0.55 * bw
ax.add_patch(Rectangle((-SW / 2, 0.0), SW, CAP, facecolor=C_NOISE_PALE,
                       edgecolor="none", zorder=4))
yb, db = 0.70 * CAP, 0.05 * CAP
ax.add_patch(Polygon([(-SW, yb - db), (SW, yb + db * 0.2),
                      (SW, yb + db * 1.2), (-SW, yb)],
                     closed=True, facecolor="white", edgecolor="none",
                     zorder=5))
ax.text(0.0, CAP * 1.04, "%d%% exactly 0" % round(100 * zero_frac),
        ha="left", va="bottom", fontsize=7.6, color=C_MUTED, zorder=6)

# ---- q95 and the 5 % beyond it ----------------------------------------------
ax.plot([q95, q95], [0.0, CAP * 0.86], color=C_NOISE, lw=1.6, zorder=6,
        solid_capstyle="butt")
ax.text(q95, CAP * 0.88, r"$q_{95}$", ha="center", va="bottom", fontsize=11,
        color=C_NOISE, zorder=6)
beyond = frac[edges[1:] > q95]
y5 = (float(beyond.max()) if beyond.size else 0.0) + 0.10 * CAP
ax.text(0.5 * (q95 + XMAX), y5, "5%", ha="center", va="bottom",
        fontsize=8.5, color=C_NOISE, zorder=6)

# ---- axis: sigma_true, schematic (only 0 marked) ----------------------------
x_left = -0.9 * bw
ax.add_patch(FancyArrowPatch((x_left, 0.0), (XMAX * 1.10, 0.0),
                             arrowstyle="-|>", mutation_scale=9, lw=1.0,
                             color=C_AXIS, shrinkA=0, shrinkB=0, zorder=7,
                             clip_on=False))
ax.text(0.0, -0.06 * CAP, "0", ha="center", va="top", fontsize=9,
        color=C_MUTED)
ax.text(XMAX * 1.10, -0.06 * CAP, r"$\hat\sigma_{\rm true}$", ha="right",
        va="top", fontsize=11, color=C_TRUE)

# ---- title --------------------------------------------------------------------
ax.text(0.5 * (x_left + XMAX * 1.10), CAP * 1.30, "Zero structure, same noise",
        ha="center", va="bottom", fontsize=TITLE_FS, weight="bold",
        color=C_TEXT)

ax.set_xlim(x_left - 0.1 * bw, XMAX * 1.2)
ax.set_ylim(-0.30 * CAP, CAP * 1.55)

# ---- crop to what is drawn ----------------------------------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
drawn = [a.get_window_extent(rend)
         for a in list(ax.texts) + list(ax.lines) + list(ax.patches)
         if a.get_visible()]
CROP = Bbox.union(drawn).transformed(fig.dpi_scale_trans.inverted())
CROP = CROP.padded(0.04)
for ext in ("png", "svg"):
    fig.savefig(ROOT / ("fig1b_null.%s" % ext), bbox_inches=CROP,
                transparent=True)
plt.close(fig)

print("benchmark: %s   %d null samples of sigma_true" % (name, N))
print("exactly zero: %.1f %%   q95 stored %.3f / recomputed %.3f   share above "
      "q95: %.1f %%" % (100 * zero_frac, q95, np.percentile(null, 95),
                       100 * np.mean(null > q95)))
print("x range to %.3f (99.5th percentile x 1.15); spike drawn at %.1f %% of "
      "its true height" % (XMAX, 100 * CAP / zero_frac))
print("saved crop %.2f x %.2f in" % (CROP.width, CROP.height))
print("Saved fig1b_null.png / .svg")
