# -*- coding: utf-8 -*-
"""Supplementary Fig. 4 data -- a picture cannot grade a collapse.

One synthetic field (single ridge, seed 51300003, profile exp), the same anchors and the same
noise draw (rng seed 13 s) at every lam of LAMS; only the transverse departure changes.  Stage
two runs as deployed: alpha_bar and sigma_alpha from the averaged profile, the window search,
rho_poly with the degree ladder, tau_min from the residual pool, rho_true_hat and the verdict.
Saved per lam: u (anchors projected on the chosen direction), yn, the polynomial at the chosen
degree (fitted on standardised u), the degree, rho, tau_min, rho_true_hat and the verdict.

Output: data/s4_borderline.npz.  A few seconds.
"""
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, TAU, DEGREES, plan_uniform, merged_run, read_interp, axial, rho_poly, window_rho, v2  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, calibrate_eps  # noqa: E402

warnings.simplefilter("ignore", np.RankWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 51300003
LAMS = (0.0, 0.010, 0.020, 0.035, 0.060, 0.120)


def ladder(u, y):
    """rho_poly's degree ladder, returning the fitted polynomial and the degree as well"""
    uu = (u - u.mean()) / max(u.std(), 1e-30)
    sy = max(float(np.std(y, ddof=1)), 1e-30)

    def at(deg):
        p = np.polyfit(uu, y, deg)
        r = y - np.polyval(p, uu)
        return math.sqrt(float(np.sum(r ** 2)) / max(y.size - deg - 1, 1)) / sy, p

    best, pb = at(DEGREES[0])
    db = DEGREES[0]
    for d in DEGREES[1:]:
        v, p = at(d)
        if v < best * 0.95:
            best, pb, db = v, p, d
        else:
            break
    return best, pb, db


nm, f, sc, a_, b_ = draw_ridge(SEED)
eps = calibrate_eps(mk(sc, f, a_, b_, 0.0), SEED)
S = plan_uniform(M_ANCH)
t_true = math.atan2(b_, a_) % math.pi
print("field seed %d: profile %s, direction %.2f deg, eps %.4g, N_MERGE %d, TAU %.2f" % (SEED, nm, math.degrees(t_true), eps, N_MERGE, TAU))
print("%7s %9s %9s %9s %6s %9s %10s" % ("lam", "rho", "tau_min", "rho_true", "deg", "verdict", "dir err"))
save = {"lams": np.array(LAMS), "profile": nm, "seed": SEED, "tau": TAU, "n_merge": N_MERGE}
for k, lam in enumerate(LAMS):
    CX, CY, HX, HY, FC, F0 = build_field(mk(sc, f, a_, b_, lam), M_ANCH, SEED)
    SFC = FC - F0[:, None]
    prof, pool, yn, _E = merged_run(SFC, F0, eps, np.random.default_rng(SEED * 13), S)    # the same noise for every lam
    ab, sa = axial(read_interp(prof, S, HX, HY))
    rp, a_star = window_rho(CX, CY, yn, ab, sa, rho_poly)
    u = math.cos(a_star) * CX + math.sin(a_star) * CY
    rho, p, deg = ladder(u, yn)
    assert abs(rho - rp) <= 1e-12 * max(rp, 1.0), (lam, rho, rp)
    tm = F.tau_min(pool, yn)
    rt = F.rho_true_hat(rho, tm)
    vd = v2(rt, tm)
    derr = abs(math.degrees((a_star - t_true + math.pi / 2) % math.pi - math.pi / 2))
    print("%7.3f %9.4f %9.4f %9.4f %6d %9s %9.3f" % (lam, rho, tm, rt, deg, vd, derr))
    save.update({"u%d" % k: u, "y%d" % k: yn, "p%d" % k: p, "deg%d" % k: deg, "rho%d" % k: rho,
                 "tau_min%d" % k: tm, "rho_true%d" % k: rt, "verdict%d" % k: vd})
out = os.path.join(HERE, "data", "s4_borderline.npz")
np.savez(out, **save)
print("saved", out)
