"""
Figure 1a — LEFT half: universe of empirical correlations, organised by the
NUMBER of independent dimensionless groups (purely structural, domain-free).

    band 1 (top)    : pi = f(pi_1)               1 group
    band 2 (middle) : pi = f(pi_1, pi_2)         2 groups   <- METHOD (focus)
    band 3 (bottom) : pi = f(pi_1, pi_2, pi_3)   3 groups

High-end / layered rendering:
  - figure-ground depth: the 2-group band is crisp, saturated, dense and
    softly GLOWING (floats forward); the 1- and 3-group context bands are
    faded, desaturated and sparse (recede).
  - per-cell value jitter so fills read as a rich texture, not flat colour.
  - multi-ring glow on the highlighted instances.

Figure1 copy (figure_noise/Figure1).  Differences from the original in
figure/methodology:
  - NOISE inside the two-input cells.  The cell's own colour is the law; a
    continuous random field shown on top of it is the noise.  Nothing is added
    to the map, so the noise cannot be misread as more correlations (red dots
    around each seed were tried and read exactly that way).
      * continuous: white noise sampled every NOISE_RES, smoothed to the
        correlation length NOISE_CORR, rescaled to unit sd.  Independent
        square grains were tried and looked fragmented.  The field runs
        straight across cell borders; each cell shows it in its own colour.
      * hue kept: a darker spot moves toward a deeper shade of the same hue,
        a lighter one toward white.  A black veil turned the pastels grey.
      * weak: NOISE_AMP 0.20.  At 0.35 the band was busy and the blue system
        dot got lost.  NOISE_CORR 0.09 was tried too and read as watercolour.
      * the 1- and many-group bands stay flat, so the texture reads as noise
        by contrast.  White edges and seed dots are drawn on top, so the laws
        stay crisp and only what is observed of them is noisy.
      * one amplitude everywhere: SCHEMATIC, no system's noise level is
        claimed.  The caption has to say so.
  - the three system dots take their own cell's hue, darker, at one fixed
    lightness (DOT_MATCH, DOT_LIGHT), so each reads as the core of its cell
    rather than a foreign colour laid on top.  It also retires terracotta and
    green here, which mean noise and REDUCIBLE in the later panels.  The
    connecting lines and box frames at assembly should use the same colours;
    the script prints them.
  - the two-input label reads Pi_o = f(Pi_1, Pi_2) + e; the other two labels
    do not.  The term marks the problem this work addresses and matches the
    only band that shows the noise texture.  It does NOT claim the other
    correlations are noise-free, and the caption must not suggest it.
  - the old in-script title is gone (the panel title is set at assembly) and
    the top margin now matches the other three sides.
  - the pastel palette is the ORIGINAL one, by decision.  It shares pale
    green / peach / grey with the verdict fills of the later panels; that is
    to be settled together with those panels.  A neutral alternative that was
    tried: band cells #cdd4dd..#dbe1e8, outer bands #e6eaee, system dots ink
    #1f2937 with a white ring.

Outputs (this folder): fig1a_universe.svg / .png
"""

import colorsys
import os
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, cKDTree

ROOT    = Path(__file__).parent
OUT_SVG = ROOT / "fig1a_universe.svg"
OUT_PNG = ROOT / "fig1a_universe.png"

# Nature's figure guide: text in Helvetica or Arial, Greek letters in the
# Symbol font.  matplotlib cannot read Symbol (every glyph comes out missing),
# so Greek letters are set upright in Times New Roman, whose serif upright
# Greek is close to Symbol's.  Latin variables are Arial italic; digits,
# brackets and operators Arial upright.  Same setup as fig1c_flow_v2.py.
plt.rcParams.update({
    "font.family":       "Arial",     # by name: 'sans-serif' went into the SVG
    "mathtext.fontset":  "custom",
    "mathtext.rm":       "Arial",
    "mathtext.it":       "Arial:italic",
    "mathtext.bf":       "Arial:bold",
    "mathtext.sf":       "Times New Roman",
    "mathtext.fallback": "stixsans",
    "svg.fonttype":      "none",
    "pdf.fonttype":      42,
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

# palette ---------------------------------------------------------------
# Each focus cell = one physics problem -> draw from a multi-hue pastel pool
# so the band reads as a *family of distinct problems*, not one blue blob.
FOCUS_PALETTE = [
    "#c7d8eb",   # soft blue
    "#ecd0c5",   # soft terra
    "#ccdfd2",   # soft green
    "#d9cde0",   # soft mauve
    "#bfdadb",   # soft teal
    "#ebd9b7",   # soft ochre
    "#ced3da",   # soft slate
    "#ecc8cf",   # soft pink
    "#dde4b3",   # soft olive
    "#c8d4e8",   # soft periwinkle
]
# Context bands: pale near-grey tints (recede but still hint at variety).
CTX_PALETTE = [
    "#e6eaee", "#ebe6e2", "#e5ebe7",
    "#eae5ed", "#e3e9ea", "#ece8de",
]
C_FOCUS_EDGE_CELL = "#ffffff"      # white edges -> clean mosaic look
C_CTX_EDGE   = "#e8ebef"
C_SEED_FOCUS = "#5d6675"
C_SEED_CTX   = "#b7bec8"
C_TEXT       = "#1f2937"
C_MUTED      = "#5d6675"
C_GLOW       = "#a9cdf2"
HL           = ["#b54f30", "#2f6fb0", "#3f7d5a"]   # mass transfer / natural convection / pipe flow

# noise inside the two-input cells (see the docstring) -------------------
NOISE_ON   = os.environ.get("NOISE", "1") == "1"
NOISE_AMP  = float(os.environ.get("NOISE_AMP", "0.20"))    # sd of the shift
NOISE_CORR = float(os.environ.get("NOISE_CORR", "0.05"))   # correlation length
NOISE_RES  = float(os.environ.get("NOISE_RES", "0.02"))    # sampling step
NOISE_SEED = int(os.environ.get("NOISE_SEED", "3"))

# system dots in their own cell's hue (see the docstring) ----------------
DOT_MATCH  = os.environ.get("DOT_MATCH", "1") == "1"
DOT_LIGHT  = float(os.environ.get("DOT_LIGHT", "0.45"))    # one lightness

rng = np.random.default_rng(7)


def jitter(hexcol, amt):
    rgb = np.array(mcolors.to_rgb(hexcol))
    return tuple(np.clip(rgb + rng.uniform(-amt, amt, 3), 0, 1))


# Voronoi finite-polygon helper ----------------------------------------
def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if all(v >= 0 for v in verts):
            new_regions.append(verts); continue
        nr = [v for v in verts if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]; t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            mid = vor.points[[p1, p2]].mean(axis=0)
            far = vor.vertices[v2] + np.sign(np.dot(mid-center, n)) * n * radius
            nr.append(len(new_vertices)); new_vertices.append(far.tolist())
        vs = np.asarray([new_vertices[v] for v in nr]); c = vs.mean(axis=0)
        nr = np.array(nr)[np.argsort(np.arctan2(vs[:, 1]-c[1], vs[:, 0]-c[0]))]
        new_regions.append(nr.tolist())
    return new_regions, np.asarray(new_vertices)


# seeds in 3 horizontal bands ------------------------------------------
BX0, BY0, BX1, BY1 = 0.4, 0.4, 9.6, 9.6
FY0, FY1 = 3.0, 7.15          # focus (2-group) band y-extent


def band(n, y0, y1):
    return np.column_stack([rng.uniform(1.0, 9.0, n), rng.uniform(y0, y1, n)])

G1 = band(12, 7.6, 9.2)       # 1 group  (context)
G2 = band(50, 3.25, 7.0)      # 2 groups (focus, dense)
G3 = band(12, 0.8, 2.4)       # 3 groups (context)
seeds = np.vstack([G1, G2, G3])
group = np.array([0]*len(G1) + [1]*len(G2) + [2]*len(G3))


def lloyd_relax(seeds, group_idx, x_bounds, y_bands, n_iter=4, grid=140):
    """Lloyd's relaxation: pull each seed toward its Voronoi-cell centroid,
    so cells become more regular and seeds sit near cell centres.
    y is clamped per-band so seeds never drift across band boundaries.

    seeds       : (N, 2)
    group_idx   : (N,) band index per seed
    x_bounds    : (x0, x1) sampling-grid x range  (= clip-rect x)
    y_bands     : list of (y_lo, y_hi) per group
    """
    seeds = seeds.copy().astype(float)
    x0, x1 = x_bounds
    y0 = min(yl for yl, _ in y_bands)
    y1 = max(yh for _, yh in y_bands)
    gx, gy = np.meshgrid(
        np.linspace(x0, x1, grid),
        np.linspace(y0, y1, grid),
    )
    pts = np.column_stack([gx.ravel(), gy.ravel()])           # (G^2, 2)

    for _ in range(n_iter):
        # nearest-seed assignment, vectorised
        d2 = ((pts[:, None, 0] - seeds[None, :, 0]) ** 2 +
              (pts[:, None, 1] - seeds[None, :, 1]) ** 2)
        nearest = np.argmin(d2, axis=1)
        new = seeds.copy()
        for i in range(len(seeds)):
            m = nearest == i
            if m.any():
                new[i] = pts[m].mean(axis=0)
        # clamp y to band (keep band membership stable)
        for i in range(len(seeds)):
            yl, yh = y_bands[group_idx[i]]
            new[i, 1] = np.clip(new[i, 1], yl + 0.15, yh - 0.15)
        seeds = new
    return seeds


# 4 Lloyd iterations -> seeds sit near cell centres, cells look regular.
seeds = lloyd_relax(
    seeds, group,
    x_bounds=(BX0, BX1),
    y_bands=[(7.6, 9.2), (3.25, 7.0), (0.8, 2.4)],
    n_iter=4, grid=140,
)

vor = Voronoi(seeds)
regions, vertices = voronoi_finite_polygons_2d(vor)

# figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.9, 5.7), dpi=220)
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(BX0 - 0.5, BX1 + 0.5)
# the same 0.5 margin on all four sides; the extra top room was for a title
ax.set_ylim(BY0 - 0.5, BY1 + 0.5)
clip_rect = Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0, transform=ax.transData)

# --- soft glow behind the focus band (concentric light rects -> "lift").
# Slightly stronger than before since the dashed frame no longer marks the band.
# Clipped to the outer rectangle so the halo does NOT bleed past the frame.
for pad, a in [(0.65, 0.09), (0.42, 0.14), (0.22, 0.19), (0.06, 0.24)]:
    glow = FancyBboxPatch(
        (BX0 + 0.05 - pad, FY0 - pad),
        (BX1 - BX0 - 0.1) + 2 * pad, (FY1 - FY0) + 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.7",
        facecolor=C_GLOW, edgecolor="none", alpha=a, zorder=0)
    ax.add_patch(glow)
    glow.set_clip_path(clip_rect)

CELL_FACE = {}        # each cell's own fill, which the noise builds on
# --- Voronoi cells: focus crisp & multi-hued, context faded ---
for idx, region in enumerate(regions):
    if group[idx] == 1:                                   # focus
        base = FOCUS_PALETTE[rng.integers(0, len(FOCUS_PALETTE))]
        face, ec, elw, al = jitter(base, 0.018), C_FOCUS_EDGE_CELL, 1.0, 1.0
    else:                                                 # context (recede)
        base = CTX_PALETTE[rng.integers(0, len(CTX_PALETTE))]
        face, ec, elw, al = jitter(base, 0.010), C_CTX_EDGE, 0.5, 0.55
    patch = Polygon(vertices[region], closed=True, facecolor=face,
                    edgecolor=ec, linewidth=elw, alpha=al, zorder=1)
    ax.add_patch(patch); patch.set_clip_path(clip_rect)
    CELL_FACE[idx] = tuple(face)

# --- noise: one continuous random field, shown in each two-input cell's colour
def deeper(rgb):
    """The same hue, darker and a little more saturated."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(l - 0.22, 0.0), min(s + 0.25, 1.0))


def dot_colour(rgb, light=0.45, sat_floor=0.50):
    """A system dot in its own cell's hue, at a fixed lightness so the three
    dots carry the same weight, and saturated enough to read as that hue
    rather than as grey."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, light, max(s, sat_floor))


if NOISE_ON:
    nx = int(round((BX1 - BX0) / NOISE_RES))
    ny = int(round((BY1 - BY0) / NOISE_RES))
    gxs = BX0 + (np.arange(nx) + 0.5) * (BX1 - BX0) / nx
    gys = BY0 + (np.arange(ny) + 0.5) * (BY1 - BY0) / ny
    GXg, GYg = np.meshgrid(gxs, gys)
    # the nearest seed IS the Voronoi cell, so this gives every sample its law
    owner = cKDTree(seeds).query(np.column_stack([GXg.ravel(),
                                                  GYg.ravel()]))[1]
    owner = owner.reshape(ny, nx)
    in_band = group[owner] == 1

    field = np.random.default_rng(NOISE_SEED).normal(0.0, 1.0, (ny, nx))
    field = gaussian_filter(field, sigma=NOISE_CORR / NOISE_RES,
                            mode="reflect")
    field = (field - field.mean()) / field.std()        # back to unit sd

    base = np.array([CELL_FACE[i] for i in range(len(seeds))])
    deep = np.array([deeper(c) for c in base])
    Cg, Dg = base[owner], deep[owner]
    tg = np.clip(NOISE_AMP * field, -1.0, 1.0)[..., None]
    col = np.where(tg < 0, Cg + (-tg) * (Dg - Cg), Cg + tg * 0.9 * (1.0 - Cg))
    rgba = np.concatenate([np.clip(col, 0, 1),
                           in_band[..., None].astype(float)], axis=2)
    noise_im = ax.imshow(rgba, extent=(BX0, BX1, BY0, BY1), origin="lower",
                         interpolation="bilinear", zorder=1.5)
    noise_im.set_clip_path(clip_rect)
    # white edges again, on top of the noise, so the laws stay crisp
    for idx, region in enumerate(regions):
        if group[idx] == 1:
            edge = Polygon(vertices[region], closed=True, facecolor="none",
                           edgecolor=C_FOCUS_EDGE_CELL, linewidth=1.0,
                           zorder=2)
            ax.add_patch(edge); edge.set_clip_path(clip_rect)

# outer boundary (plain rectangle, matches the square clip -> no corner spill).
# The previous dashed blue frame around the focus band was dropped — the colour
# contrast (saturated focus vs faded context + glow) already defines the band.
ax.add_patch(Rectangle((BX0, BY0), BX1-BX0, BY1-BY0,
             facecolor="none", edgecolor="#aab3bf", linewidth=1.4, zorder=4))

# --- seeds: context faint+small, focus crisp ---
ctx = group != 1
ax.scatter(seeds[ctx, 0], seeds[ctx, 1], s=6, color=C_SEED_CTX,
           alpha=0.55, zorder=3, edgecolor="none")
ax.scatter(seeds[~ctx, 0], seeds[~ctx, 1], s=11, color=C_SEED_FOCUS,
           zorder=3, edgecolor="none")

# --- highlight 3 instances in the focus band (multi-ring glow) ---
g2 = np.where(group == 1)[0]
g2 = g2[np.argsort(seeds[g2, 0])]
# ---- positions of the two highlighted instances ----
# Set the (x, y) you want for each; the marker snaps to the nearest
# correlation (seed) in the 2-group band.
#   coordinate system: box spans x, y in [0.4, 9.6];
#   the 2-group (focus) band is roughly  y in [3.25, 7.0].
#   bigger x -> right,  bigger y -> up.
# mass transfer moved one cell to the right, onto the pink cell (its seed sits
# at 6.76, 6.85): with each dot taking its own cell's hue, the first cell gave
# almost the same blue as the natural-convection dot (#394aac against #3958ac)
HL_TARGETS = [(6.76, 6.85), (8.0, 6.0), (4.5, 4.0)]   # mass transfer / natural convection / pipe flow
hl_ids = []
for tx, ty in HL_TARGETS:
    d2 = np.sum((seeds[g2] - np.array([tx, ty])) ** 2, axis=1)
    hl_ids.append(int(g2[int(np.argmin(d2))]))
# ---- highlight marker + glow tuning knobs (boosted for contrast vs pastel) ----
CORE_SIZE  = 70     # solid coloured dot size  (↑ from 50)
GLOW_SIZE  = 100     # outermost glow size  (bigger -> larger halo)
GLOW_ALPHA = 0.42    # innermost glow opacity (bigger -> denser/stronger)
GLOW_RINGS = 5       # number of concentric rings (more -> smoother)

# each dot takes its OWN cell's hue; DOT_MATCH=0 brings back the fixed HL
hl_col = ([dot_colour(CELL_FACE[sid], DOT_LIGHT) for sid in hl_ids]
          if DOT_MATCH else list(HL))
for k, sid in enumerate(hl_ids):
    px, py = seeds[sid]
    # soft coloured halo (background tint, gives the "glow" feel)
    for i in range(GLOW_RINGS):
        t = i / max(GLOW_RINGS - 1, 1)          # 0 = outer, 1 = inner
        s = GLOW_SIZE * (1 - 0.62 * t)          # outer big -> inner small
        a = GLOW_ALPHA * (0.35 + 0.65 * t)      # outer faint -> inner strong
        ax.scatter([px], [py], s=s, color=hl_col[k], alpha=a,
                   zorder=5, edgecolor="none")
    # white pad so the dot floats above ANY pastel cell
    ax.scatter([px], [py], s=CORE_SIZE * 1.7, color="white", zorder=6,
               edgecolor="none")
    # coloured core with a thin dark outline -> high contrast against pastels
    ax.scatter([px], [py], s=CORE_SIZE, color=hl_col[k], zorder=7,
               edgecolor="#1a1a1a", linewidth=1.0)

# --- band labels + method emphasis ---
def lab(x, y, txt, focus=False):
    ax.text(x, y, txt, ha="left", va="center",
            fontsize=(13 if focus else 10.0),
            weight=("bold" if focus else "normal"),
            color=(C_TEXT if focus else C_MUTED), zorder=8,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none",
                      boxstyle="round,pad=0.28"))

lab(0.7, 9.1, r"$\Pi_o = f(\Pi_1)$")
# the noise term only on the band that carries the noise texture, so the label
# and the picture say the same thing.  e, not epsilon: e is the error itself
# (brief: y_obs = y + e, Var(e) = epsilon squared), and epsilon is already the
# roughness in the pipe-flow box.
# no "\ " after the commas: with Arial it left a gap wider than a word space;
# mathtext's own comma spacing matches panel c's max(tau_1, q95)
lab(0.7, 7.6, r"$\Pi_o = f(\Pi_1, \Pi_2)" + (r" + e" if NOISE_ON else "")
    + "$", focus=True)
# \ldots, not \cdots: Arial has the baseline ellipsis, and a centred one came
# from STIX, which then showed up as a third font in the SVG
lab(0.7, 0.8, r"$\Pi_o = f(\Pi_1, \Pi_2, \Pi_3, \ldots)$")

# no title: the panel title is set at assembly ("Two-input dimensionless
# response maps"); an in-script title came back on every re-render

plt.savefig(OUT_SVG, format="svg", bbox_inches="tight",
            transparent=True, pad_inches=0.04)
plt.savefig(OUT_PNG, format="png", dpi=220, bbox_inches="tight",
            transparent=True, pad_inches=0.04)
print(f"Saved {OUT_SVG}")
print(f"Saved {OUT_PNG}")
print(f"highlight seed ids = {hl_ids}")
print("system dots (mass transfer, natural convection, pipe flow): %s"
      % ", ".join(mcolors.to_hex(c) for c in hl_col))
if NOISE_ON:
    lag = np.corrcoef(field[:, :-1].ravel(), field[:, 1:].ravel())[0, 1]
    print("noise: %d x %d samples, correlation length %.3f, neighbour "
          "correlation %.2f, shift sd %.2f, %.0f%% of samples in the band"
          % (nx, ny, NOISE_CORR, lag, NOISE_AMP, 100 * in_band.mean()))
else:
    print("noise: off")
