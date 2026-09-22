# -*- coding: utf-8 -*-
"""Shared pieces of the figure scripts.

Everything a panel needs besides its own drawing: the font scheme (Arial 7 pt, Greek upright),
the colours, the benchmark data, and stage two recomputed with the deployed functions.

Data: benchmarks/data/fig2_method_data.npz holds ONE realisation of the protocol per regime
(40 anchors, 6 repeats, 15 equispaced angles; benchmarks/fig2_data.py).  Stage-one numbers are
that realisation's; stage two is recomputed here from its anchors and averaged responses with
framework.core (window search, rho_poly, tau_min, rho_true_hat, verdicts).  The verdict modes
over 8 realisations (benchmarks/run_benchmarks.py) are in framework_pipeline_v4.npz;
compare_published() prints the comparison.
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from framework import core                                                # noqa: E402
from framework.core import DEGREES, K_WIN, N_WIN, M_ANCH, TAU, TAU1, window_rho, v1, v2, joint, STAGE2_NAME, JOINT_NAME  # noqa: E402,F401
from framework.benchmarks import resp_pipe, resp_pipe_lam, resp_bio, resp_micro, resp_buck  # noqa: E402,F401

BENCH = ROOT.parent / "benchmarks" / "data"
DATA = BENCH / "fig2_method_data.npz"
PUBLISHED = BENCH / "framework_pipeline_v4.npz"
CACHE = ROOT / "fig2_fields_cache.npz"

# ---- fonts: the Figure 1 scheme (Arial; Greek upright in Times New Roman,
#      standing in for the Symbol font the Nature guide asks for) ---------------
mpl.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.fallback": "stixsans",
    "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    # 1 pt: the thinnest line Nature Communications allows in a final figure
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 400,
})

GREEK = r"(alpha|sigma|tau|rho|mu|varepsilon|varphi|phi|kappa|lambda|Delta|pi|Pi)"


def upright_greek(s):
    """Route every Greek letter of a math string to mathtext.sf, upright."""
    s = re.sub(r"\\(hat|bar)\\" + GREEK + r"(?![A-Za-z])",
               r"\\\1{\\mathsf{\\\2}}", s)
    return re.sub(r"(?<!\\mathsf\{)\\" + GREEK + r"(?![A-Za-z])",
                  r"{\\mathsf{\\\1}}", s)


# applied to every text as it is set, so no label keeps an italic Arial Greek
_set_text = Text.set_text
Text.set_text = lambda self, s: _set_text(
    self, upright_greek(s) if isinstance(s, str) and "$" in s else s)

# ---- colours: Figure 1's grammar ------------------------------------------------
C_TEXT, C_MUTED = "#1f2937", "#5d6675"
C_DIR = "#33305f"                     # what was measured / read
C_LAV = "#dcd9ec"                     # its pale fill
C_BOX = "#8b939e"
FIELD = LinearSegmentedColormap.from_list(
    "field", ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"])
POSITION = LinearSegmentedColormap.from_list("position", ["#d9d6ea", C_DIR])
VERDICT = {"REDUCIBLE": ("#dceeed", "#86c2bb", "#1d6660"),
           "NOT-RED": ("#f6ecd5", "#d8b56b", "#85580f"),
           "UNRESOLVED": ("#eceef0", "#c4cbd2", "#5d6675")}


# ---- stage two: framework.core.rho_poly, also reporting the degree ---------------------
def rho_poly_deg(u, y):
    """framework.core.rho_poly line by line, also returning the degree it settled on;
    stages() asserts it equal to core.rho_poly."""
    def at(deg):
        uu = (u - u.mean()) / max(u.std(), 1e-30)
        r = y - np.polyval(np.polyfit(uu, y, deg), uu)
        return math.sqrt(float(np.sum(r ** 2)) / max(y.size - deg - 1, 1)) / \
            max(float(np.std(y, ddof=1)), 1e-30)
    best, best_d = at(DEGREES[0]), DEGREES[0]
    for d in DEGREES[1:]:
        v = at(d)
        if v < best * 0.95:
            best, best_d = v, d
        else:
            break
    return best, best_d


# ---- data ----------------------------------------------------------------------------
def load():
    return np.load(DATA, allow_pickle=True)


def box(d, u):
    """The regime's window, (x0, x1, y0, y1) in ln units."""
    return tuple(float(v) for v in d[u + "_box"])


def angle_dev_deg(ang, abar):
    """Axial deviation from the mean direction, wrapped to [-90, 90) degrees."""
    return np.degrees((np.asarray(ang) - abar + np.pi / 2) % np.pi - np.pi / 2)


def stages(d, u):
    """Stage one as recorded for this realisation; stage two recomputed."""
    p = u + "_"
    CX, CY, yn = d[p + "CX"], d[p + "CY"], d[p + "yn"]
    ab, sa = float(d[p + "abar"]), float(d[p + "sa"])
    st, q95 = float(d[p + "st"]), float(d[p + "q95"])
    rho, a_star = window_rho(CX, CY, yn, ab, sa, core.rho_poly)
    uproj = math.cos(a_star) * CX + math.sin(a_star) * CY
    rho_d, deg = rho_poly_deg(uproj, yn)
    assert abs(rho_d - rho) <= 1e-12 * max(rho, 1.0)
    s_y = float(np.std(yn, ddof=1))
    # yn is the mean of N_MERGE repeats, so its noise is eps_pool / sqrt(N_MERGE): tau_min =
    # eps_hat / (sqrt(n) s_y), as framework.core.tau_min with eps_hat = eps_pool of the run
    tmin = float(d[p + "eps_pool"]) / (s_y * math.sqrt(int(d["N_MERGE"])))
    rt = core.rho_true_hat(rho, tmin)
    c1, c2 = v1(st, TAU1, q95), v2(rt, tmin)
    return dict(name=str(d[p + "name"]), box=box(d, u), CX=CX, CY=CY, yn=yn,
                ang=d[p + "ang"], abar=ab, sa=sa, sn=float(d[p + "sn"]), st=st,
                q95=q95, c1=c1, rho=rho, a_star=a_star, u=uproj, deg=deg,
                tmin=tmin, rt=rt, c2=c2, jt=joint(c1, c2))


def fields(d, resp_of, n):
    """The response on an n x n grid over each regime's window, cached."""
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    fresh = False
    for u, resp in resp_of.items():
        if (u + "_Z") in cache and cache[u + "_Z"].shape == (n, n):
            continue
        x0, x1, y0, y1 = box(d, u)
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        cache[u + "_xs"], cache[u + "_ys"] = xs, ys
        cache[u + "_Z"] = np.array([[resp(x, y) for x in xs] for y in ys])
        fresh = True
    if fresh:
        np.savez(CACHE, **cache)
    return cache


def compare_published(S):
    pub = np.load(PUBLISHED, allow_pickle=True)
    units = list(pub["units"])
    print("%-24s %6s %6s %-10s %7s %4s %7s %6s %-10s %-10s | published joint / rho_poly"
          % ("regime", "st", "q95", "stage 1", "rho", "deg", "a* deg", "tmin",
             "stage 2", "joint"))
    for u, s in S.items():
        i = units.index(s["name"])
        print("%-24s %6.2f %6.3f %-10s %7.4f %4d %7.1f %6.4f %-10s %-10s | %s / %.4f"
              % (s["name"], s["st"], s["q95"], s["c1"], s["rho"], s["deg"],
                 math.degrees(s["a_star"]) % 180, s["tmin"], STAGE2_NAME[s["c2"]],
                 JOINT_NAME[s["jt"]], str(pub["joint"][i]),
                 float(pub["rho_poly"][i])))
