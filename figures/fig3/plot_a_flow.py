"""
plot_a_flow.py —— panel a (逻辑流版):
  顶部: 真实显微时序 (0/12/24h, CC BY).
  下方三段逻辑流:
   ① 受伤单层 + 两个竞争过程 (迁移 D_n 橙 vs 增殖 r 绿) + L, n0->K
   ② 闭合曲线: 缝宽随时间->0, 定义响应 T_c
   ③ 量纲塌缩: {D_n,r,L}->Π1=D_n/(rL²)=迁移/增殖,  {n0,K}->Π2=n0/K;  T_c·r=f(Π1,Π2)
颜色把 #2(5量->2群) 与 #3(迁移vs增殖竞争) 串起来.

Figure 3 副本 (figure_noise/Figure3):
  * 字体换成 Figure 1/2 的方案: Arial 正文, 希腊字母用正体 Times New Roman
    (导入 Figure2/fig2_common.py 设定); 原来是 serif + STIX
  * 按最终尺寸画: 180 mm 宽, 占满图 3 整个宽度. 所有文字 7 pt (Nature 上限);
    原来 6.5-10.5 pt, 整体缩放无法同时满足 5-7 pt, 层次改靠粗体和颜色
  * 最细的线 1 pt (NC 要求); 原来有 0.6-0.9 pt 的虚线和引线
  * 数据范围按内容收紧, 四周留 1 mm
  * 信息框按内容定大小 (7 pt 下原来的框太空), 左边离 cell monolayer 标注 5 mm;
    其余版式和内容不动
  * 显微图仍从 v7_biology/figures 读取, 不复制
输出: fig3_a_flow.png/.svg (本目录)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
import matplotlib.patheffects as pe

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 显微图留在原处, 输出写到本目录
IMG_DIR = os.path.join(SCRIPT_DIR, 'img')                # the two micrographs
# 字体: Figure 1/2 的方案 (Arial 正文 + 正体 Times New Roman 希腊字母), 导入即生效
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
import common  # noqa: E402,F401  fonts

TXT, MUT = '#1f2937', '#5d6675'
CYTO, NUC, CEDGE = '#cfe1ec', '#46789c', '#6a9bbd'
MIG, GRO, DEN = '#c0603a', '#d81b60', '#3f6fa0'      # 迁移橙 / 增殖玫红 / 密度蓝
BG, BGED = '#f6f7f9', '#cdd3da'

# 最终尺寸: 180 mm 宽. 坐标轴铺满画布 (等比例), 数据范围在文末按内容收紧
W_IN = 180 / 25.4
FS = 7        # 所有文字 7 pt (Nature 上限)
LW = 1.0      # 最细的线 1 pt (NC 要求)
fig = plt.figure(figsize=(W_IN, W_IN * (8.8 - 4.55) / (12.95 + 0.1)), dpi=400)  # 按保存的 dpi 量文字
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.1, 12.95); ax.set_ylim(4.55, 8.8); ax.set_aspect('equal'); ax.axis('off')

# ---------- 示意图: 0-12h 真实图 + 0h 放大标注块 ----------
# 0-12h 竖排图 (展示随时间闭合)
z012 = plt.imread(os.path.join(IMG_DIR, '0-12h.png'))
ph, pw = z012.shape[0], z012.shape[1]
PX0, PY1, PW = 0.45, 8.15, 2.2
PH = PW * ph / pw
PX1, PY0 = PX0 + PW, PY1 - PH
ax.imshow(z012, extent=(PX0, PX1, PY0, PY1), aspect='auto', zorder=3)
ax.add_patch(Rectangle((PX0, PY0), PW, PH, fill=False, edgecolor=CEDGE, lw=1.0, zorder=6))
h0 = 0.448 * PH                          # 0h 框高度 (按图比例; 0.448 来自手动调好的比例)
ax.add_patch(Rectangle((PX0, PY1 - h0), PW, h0, fill=False, edgecolor=MIG,
                       lw=1.0, ls=(0, (3, 2)), zorder=7))                # 框出 0h
# growth r (方案A + 对比): 同一位置 0h vs 12h 各画一个圈, 体现"细胞变多"
cmpx = PX0 + 0.70 * PW                          # 0h=空缝, 12h=右前沿已推过(细胞填满) -> 对比最强
c0y, c1y = PY0 + 0.72 * PH, PY0 + 0.26 * PH    # 0h(上) / 12h(下) 同一 x
ax.plot([cmpx, cmpx], [c0y - 0.16, c1y + 0.16], color=GRO, lw=LW,
        ls=(0, (2, 2)), alpha=0.85, zorder=7)   # 虚线: 同一位置
for cy in (c0y, c1y):
    ax.add_patch(Circle((cmpx, cy), 0.16, fill=False, edgecolor=GRO, lw=1.7, zorder=8))
_fm = [pe.withStroke(linewidth=1.4, foreground='white')]     # 白描边, 在紫灰底上可读
ax.text(cmpx - 0.21, c0y, 'few', fontsize=FS, color=GRO, ha='right', va='center', style='italic', path_effects=_fm)
ax.text(cmpx - 0.21, c1y, 'many', fontsize=FS, color=GRO, ha='right', va='center', style='italic', path_effects=_fm)
# 标签居中于两圈正下方, 竖直引线
ax.annotate('proliferation $r$', xy=(cmpx, c1y - 0.18), xytext=(cmpx, PY0 - 0.16),
            fontsize=FS, color=GRO, ha='center', va='top', style='italic', fontweight='bold',
            arrowprops=dict(arrowstyle='-', lw=LW, color=GRO))

# 0h 放大块 (承载物理量标注)
z0 = plt.imread(os.path.join(IMG_DIR, '0h.png'))
zh, zw = z0.shape[0], z0.shape[1]
CX0, CY1, CW2 = 3.15, 7.95, 2.7
CH2 = CW2 * zh / zw
CX1, CY0 = CX0 + CW2, CY1 - CH2
ax.imshow(z0, extent=(CX0, CX1, CY0, CY1), aspect='auto', zorder=3)
ax.add_patch(Rectangle((CX0, CY0), CW2, CH2, fill=False, edgecolor=MIG, lw=1.2, zorder=6))
# zoom 连线: 0h 框 -> 放大块
ax.plot([PX1, CX0], [PY1, CY1], color=MIG, lw=LW, ls=(0, (3, 2)), alpha=0.6, zorder=5)
ax.plot([PX1, CX0], [PY1 - h0, CY0], color=MIG, lw=LW, ls=(0, (3, 2)), alpha=0.6, zorder=5)
# 放大块上标注
gxl = CX0 + 0.16 * CW2; gxr = CX0 + 0.79 * CW2; gxc = (gxl + gxr) / 2
for yy in (CY0 + 0.34 * CH2, CY0 + 0.66 * CH2):
    ax.add_patch(FancyArrowPatch((gxl + 0.03, yy), (gxl + 0.40, yy), arrowstyle='-|>',
                 mutation_scale=9, lw=1.5, color=MIG, zorder=8))
    ax.add_patch(FancyArrowPatch((gxr - 0.03, yy), (gxr - 0.40, yy), arrowstyle='-|>',
                 mutation_scale=9, lw=1.5, color=MIG, zorder=8))
ax.text(gxc, CY1 + 0.10, r'migration  $D_n$', fontsize=FS, color=MIG, ha='center', va='bottom',
        style='italic', fontweight='bold')
ax.add_patch(FancyArrowPatch((gxl, CY0 - 0.14), (gxr, CY0 - 0.14), arrowstyle='<|-|>',
             mutation_scale=8, lw=1.2, color=TXT, zorder=8))
# L is the FULL wound width, not the half-width: the solver puts the gap at
# |x/L| < 0.5, so the gap spans one length unit.  Pi_1 = D_n/(r L^2) follows.
ax.text(gxc, CY0 - 0.27, r'wound width $= L$', fontsize=FS, color=TXT, ha='center', va='top')
# density: 蓝圈圈出致密细胞片 + 标注 (定义在右侧信息框)
dcx, dcy = gxr + 0.24, CY0 + 0.52 * CH2
ax.add_patch(Circle((dcx, dcy), 0.08, fill=False, edgecolor=DEN, lw=1.0, zorder=8))
density = ax.annotate('cell monolayer\n(density $n_0$)', xy=(dcx + 0.07, dcy),
            xytext=(CX1 + 0.25, CY0 + 0.40 * CH2), fontsize=FS, color=DEN, ha='left', va='center',
            linespacing=1.3, arrowprops=dict(arrowstyle='-', lw=LW, color=DEN))

# ---------- 信息框 (B: 全符号定义, 自解释; 仿微流控 panel a) ----------
# 框按内容定大小: 文字是绝对尺寸, 所以边距/列距/行距按英寸给, 由 place_box() 在文末
# 收紧数据范围时反复摆放; 框左边离 cell monolayer 标注 BOX_GAP_IN (先取两块显微图的
# 间距, 9 mm, 离得太远)
BYC = 6.47                                  # 与左/中两栏内容垂直居中
BOX_ROWS = ((r'$T_c\, r \;=\; f(\Pi_1,\ \Pi_2)$', 'closure response'),
            (r'$\Pi_1 = D_n/(r L^{2})$', 'migration / proliferation'),
            (r'$\Pi_2 = n_0/K$', 'initial confluence'))
BOX_GAP_IN = 5.0 / 25.4
PAD_X_IN, PAD_Y_IN = 2.5 / 25.4, 2.0 / 25.4
COL_GAP_IN = 3.0 / 25.4                     # 公式列到说明列
PITCH_IN = 12 / 72                          # 行距 12 pt
box = FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.0,rounding_size=0.13',
                     facecolor='white', edgecolor=MUT, lw=LW, zorder=10)
ax.add_patch(box)
formulas = [ax.text(0, 0, f, fontsize=FS, color=TXT, ha='left', va='center', zorder=11)
            for f, _ in BOX_ROWS]
notes = [ax.text(0, 0, n, fontsize=FS, color=MUT, ha='left', va='center', zorder=11)
         for _, n in BOX_ROWS]


def place_box():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    per_in = (ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_size_inches()[0]   # 数据单位/英寸
    size = [(t.get_window_extent(rend).width / fig.dpi, t.get_window_extent(rend).height / fig.dpi)
            for t in formulas + notes]
    fw = max(w for w, _ in size[:3])
    nw = max(w for w, _ in size[3:])
    rh = max(h for _, h in size)
    x0 = (ax.transData.inverted().transform((density.get_window_extent(rend).x1, 0))[0]
          + BOX_GAP_IN * per_in)
    xf = x0 + PAD_X_IN * per_in
    xn = xf + (fw + COL_GAP_IN) * per_in
    for i, (tf, tn) in enumerate(zip(formulas, notes)):
        y = BYC + (1 - i) * PITCH_IN * per_in
        tf.set_position((xf, y)); tn.set_position((xn, y))
    half = (PITCH_IN + rh / 2 + PAD_Y_IN) * per_in
    box.set_bounds(x0, BYC - half, (2 * PAD_X_IN + fw + COL_GAP_IN + nw) * per_in, 2 * half)

# ---------- 按内容收紧数据范围: 画布保持 180 mm 宽, 四周留 1 mm ----------
# 文字大小是绝对的 (pt), 数据范围一变, 文字在数据坐标里占的大小也跟着变, 所以迭代到稳定
MARGIN_IN = 1.0 / 25.4
for _ in range(10):
    place_box()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ext = [a.get_window_extent(rend) for a in ax.images + ax.patches + ax.lines + ax.texts]
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [(min(e.x0 for e in ext), min(e.y0 for e in ext)),
         (max(e.x1 for e in ext), max(e.y1 for e in ext))])
    m = MARGIN_IN * (x1 - x0) / (W_IN - 2 * MARGIN_IN)       # 1 mm in data units once refit
    new = (x0 - m, x1 + m, y0 - m, y1 + m)
    old = ax.get_xlim() + ax.get_ylim()
    ax.set_xlim(new[0], new[1]); ax.set_ylim(new[2], new[3])
    fig.set_size_inches(W_IN, W_IN * (new[3] - new[2]) / (new[1] - new[0]))
    if max(abs(a - b) for a, b in zip(new, old)) < 1e-4:
        break
place_box()                                  # 按最终范围再摆一次 (已收敛时不变)

out = os.path.join(SCRIPT_DIR, 'fig3_a_flow')
fig.savefig(out + '.png', dpi=400, transparent=True)
fig.savefig(out + '.svg', transparent=True)
w_mm, h_mm = fig.get_size_inches() * 25.4
mm_per = w_mm / (ax.get_xlim()[1] - ax.get_xlim()[0])
plt.close(fig)
print(f'saved {out}.png/.svg  {w_mm:.1f} x {h_mm:.1f} mm; info box {box.get_width() * mm_per:.1f} x '
      f'{box.get_height() * mm_per:.1f} mm, {BOX_GAP_IN * 25.4:.1f} mm right of the label; '
      f'micrographs {(CX0 - PX1) * mm_per:.1f} mm apart')
