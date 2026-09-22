# -*- coding: utf-8 -*-
"""Supplementary Fig. 3 check -- is the low-noise upper tail of rho / tau_min caused by the window?

On the null fields of s3_floor_calibration.py (same seeds and noise draws; the deployed rho is
asserted equal to that file), rho is also evaluated at the true ridge direction and with the
window half-width floored at W_MIN_DEG, and the stage-one direction error is compared with the
half-width.  Prints the tail statistics by noise level and profile (Supplementary Note 2).

Env: N_NULL (800), W_MIN_DEG (0.5).  Output: data/s3_window_check_<N_NULL>.npz.  About 15 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, K_WIN, N_WIN, plan_uniform, merged_run, read_interp, axial, rho_poly, window_rho  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, calibrate_eps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_NULL = int(os.environ.get("N_NULL", 800))
W_MIN_DEG = float(os.environ.get("W_MIN_DEG", 0.5))
LEVELS = (1.0, 4.0, 0.25)                       # the order of s3_floor_calibration.py
COLS = ("field", "eps_mult", "tau_min", "rho_dep", "rho_floorwin", "rho_truth",
        "rho_truth_clean", "e1_deg", "half_deg", "err_dep_deg", "err_floorwin_deg", "sigma_alpha")


def axial_err_deg(a, b):
    return abs(math.degrees((a - b + math.pi / 2) % math.pi - math.pi / 2))


def window_floor(CX, CY, y, ab, sa):
    """window_rho's search, with the half-width floored at W_MIN_DEG"""
    half = max(K_WIN * math.radians(sa) / math.sqrt(M_ANCH), math.radians(W_MIN_DEG))
    return min((rho_poly(math.cos(a) * CX + math.sin(a) * CY, y), a) for a in np.linspace(ab - half, ab + half, N_WIN))


print("window check: N_NULL %d, levels %s, N_MERGE %d, half-width floor %.2f deg" % (N_NULL, LEVELS, N_MERGE, W_MIN_DEG), flush=True)
rows, prof = [], []
t0 = time.time()
for j in range(N_NULL):
    seed = 61300000 + j
    nm, f, sc, a_, b_ = draw_ridge(seed)
    resp = mk(sc, f, a_, b_, 0.0)
    eps0 = calibrate_eps(resp, seed)
    if not np.isfinite(eps0):
        continue
    CX, CY, HX, HY, FC, F0 = build_field(resp, M_ANCH, seed)
    if not np.all(np.isfinite(FC)):
        continue
    SFC = FC - F0[:, None]
    S = plan_uniform(M_ANCH)
    t_true = math.atan2(b_, a_) % math.pi
    u_true = math.cos(t_true) * CX + math.sin(t_true) * CY
    rho_truth_clean = rho_poly(u_true, F0)
    for mxx in LEVELS:
        eps = mxx * eps0
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))
        prof_, pool, yn, _E = merged_run(SFC, F0, eps, rg, S)
        ab, sa = axial(read_interp(prof_, S, HX, HY))
        rp_dep, a_dep = window_rho(CX, CY, yn, ab, sa, rho_poly)          # deployed
        rp_fw, a_fw = window_floor(CX, CY, yn, ab, sa)                    # floored window
        rp_truth = rho_poly(u_true, yn)                                    # oracle direction
        tm = F.tau_min(pool, yn)
        half = math.degrees(K_WIN * math.radians(sa) / math.sqrt(M_ANCH))
        rows.append((j, mxx, tm, rp_dep, rp_fw, rp_truth, rho_truth_clean, axial_err_deg(ab, t_true), half,
                     axial_err_deg(a_dep, t_true), axial_err_deg(a_fw, t_true), sa))
        prof.append(nm)
    if (j + 1) in (1, 5) or (j + 1) % 100 == 0:
        print("  field %4d  runs %5d  %.0f s" % (j + 1, len(rows), time.time() - t0), flush=True)

A = np.array(rows, float)
P = np.array(prof)
c = {k: i for i, k in enumerate(COLS)}
print("\n%d runs in %.0f s" % (len(A), time.time() - t0))

ref_path = os.path.join(HERE, "data", "s3_floor_800_200.npz")
if os.path.exists(ref_path):
    R = np.load(ref_path, allow_pickle=True)
    rc = {str(k): i for i, k in enumerate(R["cols"])}
    Rn = R["rows"][R["rows"][:, rc["kind"]] == 0]
    key = {(int(r[rc["field"]]), float(r[rc["eps_mult"]])): r[rc["rho"]] for r in Rn}
    diffs = [abs(key[(int(r[c["field"]]), float(r[c["eps_mult"]]))] - r[c["rho_dep"]])
             for r in A if (int(r[c["field"]]), float(r[c["eps_mult"]])) in key]
    print("wiring: deployed rho against s3_floor_800_200.npz, %d runs, max |diff| %.2e" % (len(diffs), max(diffs) if diffs else float("nan")))


def summary(label, ratio, sel):
    x = ratio[sel]
    return ("%-26s median %.2f  75th %.2f  95th %.2f  share > 1.5 %5.1f%%"
            % (label, np.median(x), np.percentile(x, 75), np.percentile(x, 95), 100 * np.mean(x > 1.5)))


for mxx in (0.25, 1.0, 4.0):
    s = A[:, c["eps_mult"]] == mxx
    tm = A[:, c["tau_min"]]
    r_dep, r_fw, r_tr = A[:, c["rho_dep"]] / tm, A[:, c["rho_floorwin"]] / tm, A[:, c["rho_truth"]] / tm
    r_cl = A[:, c["rho_truth_clean"]] / tm
    outside = A[:, c["e1_deg"]] > A[:, c["half_deg"]]
    print("\n== eps x%g (n %d)" % (mxx, s.sum()))
    print("   " + summary("deployed window", r_dep, s))
    print("   " + summary("true direction", r_tr, s))
    print("   " + summary("window floored at %.1f deg" % W_MIN_DEG, r_fw, s))
    print("   stage-one direction error median %.3f deg; true direction outside the window in %.1f%% of runs"
          % (np.median(A[s, c["e1_deg"]]), 100 * np.mean(outside[s])))
    tail = s & (r_dep > 1.5)
    if tail.any():
        print("   of the %d runs with deployed ratio > 1.5, true direction outside the window in %d (%.0f%%)"
              % (tail.sum(), np.sum(outside & tail), 100 * np.mean(outside[tail])))
    print("   by profile (median ratio): deployed / true direction / floored window / noiseless misfit at truth")
    for nm in sorted(set(P[s])):
        k = s & (P == nm)
        print("      %-9s n %3d   %.2f / %.2f / %.2f / %.2f   outside window %4.1f%%"
              % (nm, k.sum(), np.median(r_dep[k]), np.median(r_tr[k]), np.median(r_fw[k]), np.median(r_cl[k]), 100 * np.mean(outside[k])))

out = os.path.join(HERE, "data", "s3_window_check_%d.npz" % N_NULL)
np.savez(out, rows=A, cols=np.array(COLS), profile=P, w_min_deg=W_MIN_DEG, n_merge=N_MERGE, m_anch=M_ANCH, k_win=K_WIN, n_win=N_WIN)
print("\nsaved", out)
