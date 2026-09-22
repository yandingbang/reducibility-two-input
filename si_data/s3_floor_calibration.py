# -*- coding: utf-8 -*-
"""Supplementary Fig. 3 data -- does tau_min describe the noise floor of the collapse residual?

Two claims:
  (a) the floor: on an exactly reducible field rho / tau_min -> 1, rho_true_hat -> 0, and the
      verdict at tau = 5% is not NOT-RED;
  (b) the estimate: on a field with a departure, rho_true_hat recovers the noise-free residual
      rho_clean (the best rho_poly over 181 directions t_j = j pi / 180 on the noiseless anchor
      responses).

Null fields: the fields of s2_null_fields.py (seeds 61300000 + j), at eps x 1 and x 4 with the
same noise draws as there, plus eps x 0.25 with its own draws.  Tilted fields: single ridge +
linear term, seeds 62300000 + j, N_TILT fields per lam of LAMS, at eps x 1 and x 4; eps0 from
the lam = 0 field.  One run = the deployed stage two on one merged run: alpha_bar and
sigma_alpha from the averaged profile set the window, yn = averaged anchor responses,
window search with rho_poly, tau_min from the residual pool, rho_true_hat, verdict v2.

Env: N_NULL (800), N_TILT (200).  Output: data/s3_floor_<N_NULL>_<N_TILT>.npz.  About 20 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, TAU, plan_uniform, merged_run, read_interp, axial, rho_poly, window_rho, v2  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, calibrate_eps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_NULL = int(os.environ.get("N_NULL", 800))
N_TILT = int(os.environ.get("N_TILT", 200))
LEVELS_NULL = (1.0, 4.0, 0.25)
LEVELS_TILT = (1.0, 4.0)
LAMS = (0.02, 0.05, 0.10, 0.20, 0.40)
N_GLOBAL = 181
COLS = ("kind", "field", "lam", "eps_mult", "eps", "s_y", "rho", "tau_min", "rho_true",
        "verdict", "rho_clean", "sigma_alpha", "dir_err")
VERDICT = {"reduce": 0.0, "not-red": 1.0, "undet": 2.0}


def rho_global(CX, CY, y):
    """noise-free collapse residual: the best rho_poly over 181 directions, both end points of [0, pi]"""
    return min(rho_poly(math.cos(t) * CX + math.sin(t) * CY, y) for t in np.linspace(0.0, math.pi, N_GLOBAL))


def run(kind, j, seed, lam, levels, rows, prof):
    nm, f, sc, a_, b_ = draw_ridge(seed)
    eps0 = calibrate_eps(mk(sc, f, a_, b_, 0.0), seed)
    if not np.isfinite(eps0):
        return
    CX, CY, HX, HY, FC, F0 = build_field(mk(sc, f, a_, b_, lam), M_ANCH, seed)
    if not np.all(np.isfinite(FC)):
        return
    SFC = FC - F0[:, None]
    S = plan_uniform(M_ANCH)
    rc = rho_global(CX, CY, F0)
    t_true = math.atan2(b_, a_) % math.pi
    for mxx in levels:
        eps = mxx * eps0
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))       # the noise draw of s2_null_fields.py at x1, x4
        prof_, pool, yn, _E = merged_run(SFC, F0, eps, rg)
        ab, sa = axial(read_interp(prof_, S, HX, HY))
        rp, a_star = window_rho(CX, CY, yn, ab, sa, rho_poly)
        tm = F.tau_min(pool, yn)
        rt = F.rho_true_hat(rp, tm)
        derr = abs(math.degrees((a_star - t_true + math.pi / 2) % math.pi - math.pi / 2))
        rows.append((kind, j, lam, mxx, eps, float(np.std(yn, ddof=1)), rp, tm, rt, VERDICT[v2(rt, tm)], rc, sa, derr))
        prof.append(nm)


print("stage-two floor: N_MERGE %d, M %d, TAU %.2f; null %d fields x %s, tilted %d x %s x %s"
      % (N_MERGE, M_ANCH, TAU, N_NULL, LEVELS_NULL, N_TILT, LAMS, LEVELS_TILT), flush=True)
rows, prof = [], []
t0 = time.time()
for j in range(N_NULL):
    run(0.0, j, 61300000 + j, 0.0, LEVELS_NULL, rows, prof)
    if (j + 1) in (1, 5) or (j + 1) % 100 == 0:
        print("  null field %4d  runs %5d  %.0f s" % (j + 1, len(rows), time.time() - t0), flush=True)
for lam in LAMS:
    for j in range(N_TILT):
        run(1.0, j, 62300000 + j, lam, LEVELS_TILT, rows, prof)
    print("  tilt lam %.2f done  runs %5d  %.0f s" % (lam, len(rows), time.time() - t0), flush=True)

A = np.array(rows, float)
P = np.array(prof)
c = {k: i for i, k in enumerate(COLS)}
print("\n%d runs in %.0f s" % (len(A), time.time() - t0))
print("\nNULL (exactly reducible): rho / tau_min, verdicts at TAU, rho_true_hat, rho_clean")
for mxx in LEVELS_NULL:
    B = A[(A[:, c["kind"]] == 0) & (A[:, c["eps_mult"]] == mxx)]
    r = B[:, c["rho"]] / B[:, c["tau_min"]]
    v = B[:, c["verdict"]]
    print("  eps x%-5g n %4d  ratio median %.3f IQR %.3f-%.3f | reduce %.1f%% not-red %.1f%% undet %.1f%%"
          " | rho_true median %.4f 90th %.4f | tau_min median %.4f | rho_clean median %.4f 90th %.4f"
          % (mxx, len(B), np.median(r), np.percentile(r, 25), np.percentile(r, 75),
             100 * np.mean(v == 0), 100 * np.mean(v == 1), 100 * np.mean(v == 2),
             np.median(B[:, c["rho_true"]]), np.percentile(B[:, c["rho_true"]], 90),
             np.median(B[:, c["tau_min"]]), np.median(B[:, c["rho_clean"]]), np.percentile(B[:, c["rho_clean"]], 90)))
print("\nTILTED: rho_true_hat against rho_clean")
for mxx in LEVELS_TILT:
    for lam in LAMS:
        B = A[(A[:, c["kind"]] == 1) & (A[:, c["eps_mult"]] == mxx) & (A[:, c["lam"]] == lam)]
        if not len(B):
            continue
        v = B[:, c["verdict"]]
        above = B[:, c["rho_clean"]] > B[:, c["tau_min"]]
        rat = B[above, c["rho_true"]] / B[above, c["rho_clean"]]
        print("  eps x%-4g lam %.2f  n %3d  rho_clean med %.4f  rho_true med %.4f  "
              "rho_true/rho_clean (rho_clean > tau_min, n %3d) med %s | reduce %.0f%% not-red %.0f%% undet %.0f%%"
              % (mxx, lam, len(B), np.median(B[:, c["rho_clean"]]), np.median(B[:, c["rho_true"]]),
                 above.sum(), ("%.3f" % np.median(rat)) if above.any() else "-",
                 100 * np.mean(v == 0), 100 * np.mean(v == 1), 100 * np.mean(v == 2)))
out = os.path.join(HERE, "data", "s3_floor_%d_%d.npz" % (N_NULL, N_TILT))
np.savez(out, rows=A, cols=np.array(COLS), profile=P, lams=np.array(LAMS), tau=TAU, n_merge=N_MERGE,
         m_anch=M_ANCH, verdict_codes=np.array(["reduce", "not-red", "undet"]))
print("saved", out)
