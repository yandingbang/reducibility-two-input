# -*- coding: utf-8 -*-
"""Figure 5 -- the eight benchmarks on one decision map.

Each benchmark sits at its distance from the two thresholds of Figure 1c:
  x  stage one: sigma_hat_true / max(tau_1, q95)
  y  stage two: rho_hat_true / max(tau, tau_min),
     rho_hat_true = sqrt(max(rho^2 - tau_min^2, 0)), the statistic stage two
     compares (v2 in framework_pipeline.py), as sigma_hat_true is for stage one
so the joint rule reads off the plane: REDUCIBLE needs both below 1 (the lower
left), NOT-RED either above 1.

The values are the realisation of the merged protocol that Figures 2-4 show
(deploy_stop/figures/fig2_method_data.npz, stage two recomputed by
Figure2/fig2_common.stages), so a number read here is the number in those figures.  Its
joint verdicts equal the modes of the in-service run (framework_pipeline_v4.npz, 8
campaigns per benchmark), where every campaign gave the same verdict (joint_frac = 1);
this is checked and printed.  The lower left corner reads REDUCIBLE only while both gates pass;
where q95 >= tau_1 the denominator of x is q95 and a point below 1 on both axes is UNRESOLVED
instead (above 1 is NOT-RED either way).  The gate margins are asserted and printed, and the
caveat goes in the caption rather than on the plot (user decision, 2026-09-17).
In all eight the noise floors lie below the tolerances
(q95 <= 2.28 deg < tau_1 = 4 deg, tau_min <= 0.0102 < tau = 0.05), so the denominators are
the tolerances and none of them is UNRESOLVED.

History.  Until 2026-09-15 the plot showed the in-service medians over the 8 campaigns,
which differ from the realisation of Figures 2-4; the author noticed the mismatch and asked
for the realisation.  Earlier on 2026-09-15 y changed from rho to rho_hat_true, the statistic
the rule compares.  On 2026-09-16 the candidate grid became SIZE = 99, so that the 15 queried
angles are exactly equispaced; every number here moved a little with it (low Re's
sigma_hat_true 0 -> 0.56 deg, buckling 0.63, microfluidic 0, wound fast 0.26), the eight
verdicts did not, and the four points that share the y = 0 line regrouped.

Both axes are symmetric-log because sigma_hat_true is exactly 0 in two benchmarks (high Re,
microfluidic) and rho_hat_true in four (low Re, high Re, microfluidic, column buckling).
x is linear inside 0.01 and y inside 0.05.  Both negative bands are drawn wider than the data
needs -- x from -0.010, y from -0.050, neither with a tick -- so that x = 0 sits 6.5 mm right
of the left spine and the y = 0 row 9.2 mm above the bottom one; that is what makes room for a
picture UNDER a point (2026-09-17, the author's idea; before it, the row had 4.1 mm and the
crowded pictures had to stand side by side with a leader).  Two pairs fall on (nearly) one
spot: high Re and microfluidic both at (0, 0), and low Re (0.140, 0) with column buckling
(0.157, 0), whose markers are 1.1 mm apart on the page.  Each pair is drawn to either side of
its spot, the left member left (high Re -1.1 mm, microfluidic +1.1; low Re -0.6, buckling
+0.6); only the drawing moves, not the values.  Marker colour is the joint verdict in
Figure 1c's colours and marker shape the system; the points that shape alone does not
identify are named.

System pictures (2026-09-15, the author's request): readers may not know rough-pipe flow or
a scratch assay by name, so a small picture of its system sits by every point, in a light
1 pt frame.  The pictures are the author's existing figures, cropped, read only:
  rough pipe       Image/pipe_clean.png, the render, cropped to the pipe
  wound healing    v7_biology/figures/12h.png, Figure 3a's t = 12 h micrograph, its time
                   label and scale bar cut away (the drawn schematic was used first)
  microfluidic     v6_microfluidics/figures/fig_micro_schematic.png, the T-junction and
                   three droplets
  column buckling  v8_buckling/figures/fig_a_schematic.png, the two columns and the
                   cross-section
Their serif labels (P, L, L_d, Q_d, the equation boxes) are whited out, since at this size
they would be about 2 pt; nothing else in the drawings changes.
Placement: above the point, and above any text the frame would touch.  Exceptions, where
points crowd: each pair on the y = 0 line is split, the pipe picture above its circle and the
other picture (microfluidic, buckling) under its own marker, so that every picture sits on the
point it belongs to and no leader is needed (the elbow-leader drawing is kept for the next
crowded pair, with LEADER empty); the two pictures under the row share one height; both pipe
names sit over their pictures, there being no room by the point itself; the three wound points
are named above the point (fast since 2026-09-15), each picture above its name; medium Re's
picture sits under its name, below the point, because intermediate's picture takes the space
above that pair.  A picture that would overlap one already placed moves right.  The key is the one of the
88 x 80 mm version; the plot is 100 mm tall and the y axis runs to 12 to make room.
Overlaps of pictures, names, points and the axes are checked and printed.
Arial 7 pt, lines of at least 1 pt, drawn at final size, white background.

Output: fig5.png / .svg, 88 mm wide
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator
from matplotlib.transforms import ScaledTranslation
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours, data, stages

MM = 1 / 25.4
FS = 7                           # every label at 7 pt, the ceiling of Nature's figure guide
SYSTEMS = (("Rough-pipe", "o", r"Rough pipe (low, medium, high $Re$)"),
           ("Wound", "s", "Wound healing (slow, intermediate, fast)"),
           ("Microfluidic", "^", "Microfluidic droplets"),
           ("Column buckling", "D", "Column buckling"))
# the points shape alone does not identify: name, offset (points), alignment
NAMED = {"Rough-pipe, medium Re": (r"medium $Re$", (0, -6), "center", "top"),
         "Wound, intermediate": ("intermediate", (0, 6), "center", "bottom"),
         "Wound, slow migration": ("slow", (0, 6), "center", "bottom"),
         # named since 2026-09-15 (the author noticed it missing): it was the one wound
         # point left to be told apart from the other two only by elimination
         "Wound, fast migration": ("fast", (0, 6), "center", "bottom")}
# a name set over its picture instead of by its point: both pipe points of the y = 0 row have
# a picture above them and another under them, so there is no room by the point itself
NAMED_OVER_PICTURE = {"Rough-pipe, high Re": r"high $Re$",
                      "Rough-pipe, low Re": r"low $Re$"}
NAME_GAP = 0.6                                              # mm, a frame to its name
POINT_S = 28                                                # marker area, pt^2
X_LIN, Y_LIN, LINSCALE = 0.01, 0.05, 0.3                    # symlog: linear inside +-X_LIN / +-Y_LIN
Y_TOP = 12.0
# the two pairs that fall on (nearly) one spot, drawn side by side; only the drawing moves
DODGE_MM = {"Rough-pipe, high Re": -1.1, "Microfluidic": 1.1,
            "Rough-pipe, low Re": -0.6, "Column buckling": 0.6}

# ---- the system pictures ------------------------------------------------------------
IMG = ROOT / "img"                                          # the four system pictures
# source, crop box, the width the boxes were read at (None: the image's own pixels),
# boxes to white out (the serif labels).  "alpha": crop to the opaque pixels
PICTURES = {
    "pipe": (Path(r"E:\Flies in Sweden\Project\Seek Path\Image\pipe_clean.png"), "alpha", None, ()),
    # the t = 12 h micrograph of Figure 3a (545 x 418 px), the bottom 20 % cut away with the
    # time label and the scale bar.  At the frame's size it showed both cell sheets and the
    # gap best: t = 0 h's gap filled the frame, t = 24 h's was a crack; the drawn schematic
    # (fig_a_schematic.png) was used first
    "wound": (IMG / "wound_12h.png", (0, 0, 545, 334), None, ()),
    "micro": (IMG / "micro_schematic.png", (560, 115, 1500, 712), 2000,
              ((855, 125, 1275, 240),            # dispersed phase Q_d (water)
               (1095, 425, 1320, 535),           # the L_d arrow and label
               (1280, 70, 1945, 460))),          # the equation box
    "buckling": (IMG / "buckling_schematic.png", (225, 170, 1100, 885), 2000,
                 ((342, 450, 395, 515),          # P over the short column
                  (768, 185, 818, 250),          # P over the slender column
                  (803, 282, 882, 822),          # the L arrow and label
                  (885, 683, 1145, 800))),       # the cross-section's labels
}
PICTURE_OF = (("Rough-pipe", "pipe"), ("Wound", "wound"), ("Microfluidic", "micro"),
              ("Column buckling", "buckling"))
FRAME_W, FRAME_H, FRAME_PAD, FRAME_GAP = 9.5, 6.2, 0.5, 0.8     # mm
FRAME_EDGE = "#c4cbd2"
# placement order, and where each picture goes relative to its point
ORDER = (("Rough-pipe, high Re", "above"), ("Microfluidic", "below"),
         ("Rough-pipe, low Re", "above"), ("Column buckling", "below"),
         ("Wound, fast migration", "above"),
         ("Wound, slow migration", "above"), ("Wound, intermediate", "above"),
         ("Rough-pipe, medium Re", "below"))
ALIGN = (("Microfluidic", "Column buckling"),)              # the two under the row, one height


def load_picture(key):
    src, box, read_at, blanks = PICTURES[key]
    im = Image.open(src)
    if box == "alpha":
        im = im.crop(im.getchannel("A").getbbox())
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
        s = 1.0 if read_at is None else im.size[0] / read_at
        draw = ImageDraw.Draw(im)
        for b in blanks:
            draw.rectangle(tuple(round(v * s) for v in b), fill=(255, 255, 255))
        im = im.crop(tuple(round(v * s) for v in box))
    longest = 420                                            # px, about 1000 dpi across 8.5 mm
    k = longest / max(im.size)
    return im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.LANCZOS)


def inter(a, b):
    """Do two (x0, y0, x1, y1) boxes overlap?"""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def main():
    # ---- the realisation of Figures 2-4, and its verdicts against the in-service modes ----
    d = C.load()
    S = [C.stages(d, "u%d" % i) for i in range(8)]
    units = [s["name"] for s in S]
    p = np.load(C.PUBLISHED, allow_pickle=True)
    assert units == [str(u) for u in p["units"]], "unit order differs from framework_pipeline_v4.npz"
    tau1, tau = float(p["tau1"]), float(p["tau"])
    x = np.array([s["st"] / max(tau1, s["q95"]) for s in S])
    y = np.array([s["rt"] / max(tau, s["tmin"]) for s in S])
    verdict = [C.JOINT_NAME[str(s["jt"])] for s in S]
    modes = [C.JOINT_NAME[str(j)] for j in p["joint"]]
    print("joint verdicts equal to the in-service modes: %d of 8 (joint_frac %s)"
          % (sum(a == b for a, b in zip(verdict, modes)), " ".join("%g" % f for f in p["joint_frac"])))
    # the lower left corner reads REDUCIBLE only while both gates pass.  Where q95 >= tau_1 the
    # denominator of x is q95 itself, and a point below 1 on both axes is UNRESOLVED instead --
    # the rule abstains rather than reduces.  Above 1 always means NOT-RED, gate or no gate
    g1 = max(s["q95"] / tau1 for s in S)
    g2 = max(s["tmin"] / tau for s in S)
    assert g1 < 1 and g2 < 1, "a noise floor reaches its tolerance: the lower left is UNRESOLVED"
    print("gates: largest q95/tau_1 %.3f (%s), largest tau_min/tau %.3f (%s); both below 1, so "
          "below 1 on both axes is REDUCIBLE here and not UNRESOLVED"
          % (g1, max(S, key=lambda s: s["q95"])["name"], g2, max(S, key=lambda s: s["tmin"])["name"]))

    W, H = 88 * MM, 100 * MM
    L, B, R, T = 0.50, 0.42, 0.10, 0.08
    AXH = H - B - T
    fig = plt.figure(figsize=(W, H), dpi=400)
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, AXH / H])
    ax.set_xscale("symlog", linthresh=X_LIN, linscale=LINSCALE)
    ax.set_yscale("symlog", linthresh=Y_LIN, linscale=LINSCALE)
    # the negative bands are wider than the data needs (2026-09-17, the author's idea): they
    # put x = 0 6.5 mm right of the left spine and y = 0 9.2 mm above the bottom one, which is
    # the room a picture needs under a point.  Neither band carries a tick
    ax.set_xlim(-0.010, 6.0)
    ax.set_ylim(-0.050, Y_TOP)

    # the two thresholds, and the regions they cut; the labels keep their offsets in mm
    ax.axvline(1.0, color=C.C_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(1.0, color=C.C_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    xf, yf = ax.transAxes.inverted().transform(ax.transData.transform((1.0, 1.0)))
    red, nred = C.VERDICT["REDUCIBLE"][2], C.VERDICT["NOT-RED"][2]
    ax.text(0.03, yf - 1.7 * MM / AXH, "REDUCIBLE\nboth stages below 1", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS, color=red, linespacing=1.3)
    ax.text(xf + 0.03, 2.7 * MM / AXH, "NOT-RED\neither stage\nabove 1", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS, color=nred, linespacing=1.3)

    # the benchmarks; the two on one spot are drawn side by side
    labels = {}
    for i, u in enumerate(units):
        marker = next(m for key, m, _ in SYSTEMS if u.startswith(key))
        col = C.VERDICT[verdict[i]][2]
        shift = ax.transData + ScaledTranslation(DODGE_MM.get(u, 0.0) * MM, 0, fig.dpi_scale_trans)
        ax.scatter(x[i], y[i], s=POINT_S, marker=marker, color=col, edgecolor="none",
                   zorder=3, clip_on=False, transform=shift)
        if u in NAMED:
            name, off, ha, va = NAMED[u]
            labels[u] = ax.annotate(name, (x[i], y[i]), xytext=off, textcoords="offset points",
                                    ha=ha, va=va, fontsize=FS, color=col)
    handles = [Line2D([], [], ls="none", marker=m, markersize=5,
                      markerfacecolor=C.C_MUTED, markeredgecolor="none", label=lab)
               for _, m, lab in SYSTEMS]
    # a light frame marks the key as a key, since it sits over the NOT-RED corner
    leg = ax.legend(handles=handles, loc="upper left", frameon=True, fancybox=True,
                    framealpha=1.0, facecolor="white", edgecolor="#c4cbd2",
                    fontsize=FS, handletextpad=0.3, borderpad=0.5,
                    borderaxespad=0.6, labelspacing=0.35)
    leg.get_frame().set_linewidth(1.0)

    ax.set_xticks([0, 0.01, 0.1, 1])
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    ax.set_yticks([0, 0.1, 1, 10])
    ax.set_yticklabels(["0", "0.1", "1", "10"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(direction="in", length=2)
    ax.set_xlabel(r"Stage one: $\hat\sigma_{\rm true}\,/\,\max(\tau_1, q_{95})$", labelpad=2)
    ax.set_ylabel(r"Stage two: $\hat\rho_{\rm true}\,/\,\max(\tau, \tau_{\min})$", labelpad=2)

    # ---- the pictures ------------------------------------------------------------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_mm = 25.4 / fig.dpi

    def box_mm(bb):
        return (bb.x0 * to_mm, bb.y0 * to_mm, bb.x1 * to_mm, bb.y1 * to_mm)

    axb = box_mm(ax.get_window_extent(rend))
    r_mark = np.sqrt(POINT_S) / 2 * 25.4 / 72                # marker half-size, mm
    pos = {}
    for i, u in enumerate(units):                            # where each point is drawn, mm
        px_, py_ = np.asarray(ax.transData.transform((x[i], y[i]))) * to_mm
        pos[u] = (px_ + DODGE_MM.get(u, 0.0), py_)
    text_boxes = [box_mm(t.get_window_extent(rend)) for t in ax.texts]

    # 1. a place for each: by the point, then past any text the frame would touch
    frames = {}
    for u, where in ORDER:
        key = next(k for prefix, k in PICTURE_OF if u.startswith(prefix))
        px_, py_ = pos[u]
        if where == "right":
            x0, y0 = px_ + r_mark + FRAME_GAP, py_ - FRAME_H / 2
        else:
            x0 = min(max(px_ - FRAME_W / 2, axb[0] + 0.5), axb[2] - 0.5 - FRAME_W)
            y0 = py_ + r_mark + FRAME_GAP if where == "above" else py_ - r_mark - FRAME_GAP - FRAME_H
        for _ in range(20):
            grown = (x0 - FRAME_GAP, y0 - FRAME_GAP, x0 + FRAME_W + FRAME_GAP, y0 + FRAME_H + FRAME_GAP)
            hit = [tb for tb in text_boxes if inter(grown, tb)]
            if not hit or where == "right":
                break
            if where == "above":
                y0 = max(tb[3] for tb in hit) + FRAME_GAP
            else:
                y0 = min(tb[1] for tb in hit) - FRAME_GAP - FRAME_H
        frames[u] = [key, x0, y0]
    # 2. pictures that share one height
    for group in ALIGN:
        top = max(frames[u][2] for u in group)
        for u in group:
            frames[u][2] = top
    # 3. move right past pictures already placed
    placed = []                                              # (unit, key, x0, y0, x1, y1)
    for u, _ in ORDER:
        key, x0, y0 = frames[u]
        for _ in range(20):
            hit = [q for q in placed if x0 < q[4] + FRAME_GAP and q[2] < x0 + FRAME_W + FRAME_GAP
                   and y0 < q[5] and q[3] < y0 + FRAME_H]
            if not hit:
                break
            x0 = max(q[4] for q in hit) + FRAME_GAP
        placed.append((u, key, x0, y0, x0 + FRAME_W, y0 + FRAME_H))

    over = fig.add_axes([0, 0, 1, 1])
    over.set_xlim(0, W * 25.4); over.set_ylim(0, H * 25.4); over.axis("off")
    pictures = {}
    for u, key, x0, y0, x1, y1 in placed:
        over.add_patch(FancyBboxPatch((x0, y0), FRAME_W, FRAME_H, boxstyle="round,pad=0,rounding_size=0.8",
                                      facecolor="white", edgecolor=FRAME_EDGE, lw=1.0, zorder=1))
        if key not in pictures:
            pictures[key] = load_picture(key)
        im = np.asarray(pictures[key])
        iax = fig.add_axes([(x0 + FRAME_PAD) / (W * 25.4), (y0 + FRAME_PAD) / (H * 25.4),
                            (FRAME_W - 2 * FRAME_PAD) / (W * 25.4), (FRAME_H - 2 * FRAME_PAD) / (H * 25.4)])
        h = im.shape[0] / im.shape[1]
        iax.imshow(im, extent=(0, 1, 0, h))
        iax.set_xlim(0, 1); iax.set_ylim(0, h)
        iax.set_aspect("equal", adjustable="datalim")
        iax.axis("off")
        if u in NAMED_OVER_PICTURE:
            over.text((x0 + x1) / 2, y1 + NAME_GAP, NAMED_OVER_PICTURE[u], ha="center", va="bottom",
                      fontsize=FS, color=C.VERDICT[verdict[units.index(u)]][2])

    # a thin elbow leader where a picture is not over its point (the author's request for
    # microfluidic): right from the marker, then up into the frame's bottom edge.  A straight
    # line grazed the corner of buckling's frame, 0.8 mm to the left of microfluidic's
    LEADER = ()                    # empty since 2026-09-17: every picture is over or under its
                                   # own point.  The drawing stays, for the next crowded pair
    LEADER_GAP, LEADER_IN = 0.5, 1.2                              # mm: from the marker; into the frame
    for u, key, x0, y0, x1, y1 in placed:
        if u in LEADER:
            qx, qy = pos[u]
            xe = x0 + LEADER_IN
            over.plot([qx + r_mark + LEADER_GAP, xe, xe], [qy, qy, y0], color=FRAME_EDGE, lw=1.0,
                      solid_joinstyle="miter", solid_capstyle="butt", zorder=0.5)
            print("leader for %s: from (%.1f, %.1f) right to x %.1f, up to the frame at y %.1f mm"
                  % (u, qx + r_mark + LEADER_GAP, qy, xe, y0))

    # ---- checks -----------------------------------------------------------------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [("key", box_mm(leg.get_window_extent(rend)))]
    texts += [(t.get_text().replace("\n", " "), box_mm(t.get_window_extent(rend)))
              for t in list(ax.texts) + list(over.texts)]
    marks = {u: (qx - r_mark, qy - r_mark, qx + r_mark, qy + r_mark) for u, (qx, qy) in pos.items()}
    problems = 0
    for i, u in enumerate(units):
        for v in units[i + 1:]:
            if inter(marks[u], marks[v]):
                problems += 1; print("  overlap: the points of %s and %s" % (u, v))
        for name, tb in texts:
            if inter(marks[u], tb):
                problems += 1; print("  overlap: the point of %s and text '%s'" % (u, name))
    for i, (u, key, *fb) in enumerate(placed):
        for v, _, *gb in placed[i + 1:]:
            if inter(fb, gb):
                problems += 1; print("  overlap: pictures of %s and %s" % (u, v))
        for name, tb in texts:
            if inter(fb, tb):
                problems += 1; print("  overlap: picture of %s and text '%s'" % (u, name))
        for v, mb in marks.items():
            if inter(fb, mb):
                problems += 1; print("  overlap: picture of %s and the point of %s" % (u, v))
        if fb[0] < axb[0] or fb[2] > axb[2] or fb[1] < axb[1] or fb[3] > axb[3]:
            problems += 1; print("  picture of %s leaves the axes" % u)
        print("%-24s %-9s frame x %5.1f-%5.1f  y %5.1f-%5.1f mm; point at (%5.1f, %5.1f)"
              % (u, key, fb[0], fb[2], fb[1], fb[3], pos[u][0], pos[u][1]))
    one_x, one_y = np.asarray(ax.transData.transform((1.0, 1.0))) * to_mm
    print("x = 1 line at %.1f mm, y = 1 line at %.1f mm; axes x %.1f-%.1f, y %.1f-%.1f mm; problems: %d"
          % (one_x, one_y, axb[0], axb[2], axb[1], axb[3], problems))

    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("fig5.%s" % ext), dpi=400, facecolor="white")
    plt.close(fig)
    print("fig5  %.0f x %.0f mm" % (W * 25.4, H * 25.4))
    for i, u in enumerate(units):
        print("%-24s st %6.3f  q95 %5.3f  x %.4f | rho %.4f  tmin %.4f  rt %.4f  y %.4f  %s"
              % (u, S[i]["st"], S[i]["q95"], x[i], S[i]["rho"], S[i]["tmin"], S[i]["rt"], y[i], verdict[i]))


if __name__ == "__main__":
    main()
