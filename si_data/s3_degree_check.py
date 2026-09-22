# -*- coding: utf-8 -*-
"""Supplementary Fig. 3 check, part two -- the degree ladder on noisy responses.

On the null fields of s3_floor_calibration.py (same seeds and noise draws), rho_poly's residual
is traced at every degree of the ladder, on the noisy and on the noiseless anchor responses at
the true ridge direction, to show where the ladder stops and what misfit it leaves
(Supplementary Note 2).  The re-implemented ladder is asserted equal to rho_poly on every run.

Env: N_NULL (800).  Output: data/s3_degree_check_<N_NULL>.npz.  About 5 min.
"""
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, DEGREES, merged_run, rho_poly  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, calibrate_eps  # noqa: E402

warnings.simplefilter("ignore", np.RankWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
N_NULL = int(os.environ.get("N_NULL", 800))
LEVELS = (1.0, 4.0, 0.25)
ND = len(DEGREES)


def trace(u, y):
    """rho_poly's residual at every degree of the ladder, and the index where it stops"""
    uu = (u - u.mean()) / max(u.std(), 1e-30)
    sy = max(float(np.std(y, ddof=1)), 1e-30)
    r = np.empty(ND)
    for i, deg in enumerate(DEGREES):
        res = y - np.polyval(np.polyfit(uu, y, deg), uu)
        r[i] = math.sqrt(float(np.sum(res ** 2)) / max(y.size - deg - 1, 1)) / sy
    stop = 0
    for i in range(1, ND):
        if r[i] < r[stop] * 0.95:
            stop = i
        else:
            break
    return r, stop


print("degree check: N_NULL %d, levels %s, DEGREES %s" % (N_NULL, LEVELS, DEGREES), flush=True)
rec = {k: [] for k in ("field", "eps_mult", "tau_min", "r_noisy", "r_clean", "stop_noisy", "stop_clean", "ratio_deployed_ladder")}
prof = []
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
    t_true = math.atan2(b_, a_) % math.pi
    u_true = math.cos(t_true) * CX + math.sin(t_true) * CY
    r_clean, stop_clean = trace(u_true, F0)
    for mxx in LEVELS:
        eps = mxx * eps0
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))
        _prof, pool, yn, _E = merged_run(SFC, F0, eps, rg)
        tm = F.tau_min(pool, yn)
        r_noisy, stop_noisy = trace(u_true, yn)
        deployed = rho_poly(u_true, yn)
        assert abs(deployed - r_noisy[stop_noisy]) <= 1e-12 * max(1.0, deployed), (j, mxx)
        for k, v in (("field", j), ("eps_mult", mxx), ("tau_min", tm), ("r_noisy", r_noisy), ("r_clean", r_clean),
                     ("stop_noisy", stop_noisy), ("stop_clean", stop_clean), ("ratio_deployed_ladder", deployed / tm)):
            rec[k].append(v)
        prof.append(nm)
    if (j + 1) in (1, 5) or (j + 1) % 100 == 0:
        print("  field %4d  runs %5d  %.0f s" % (j + 1, len(prof), time.time() - t0), flush=True)

R = {k: np.array(v) for k, v in rec.items()}
P = np.array(prof)
print("\n%d runs in %.0f s; the re-implemented ladder matched rho_poly on every run" % (len(P), time.time() - t0))

wc_path = os.path.join(HERE, "data", "s3_window_check_800.npz")
if os.path.exists(wc_path):
    W = np.load(wc_path, allow_pickle=True)
    wc = {str(k): i for i, k in enumerate(W["cols"])}
    key = {(int(r[wc["field"]]), float(r[wc["eps_mult"]])): r[wc["rho_truth"]] / r[wc["tau_min"]] for r in W["rows"]}
    d = [abs(key[(int(f_), float(m_))] - v) for f_, m_, v in zip(R["field"], R["eps_mult"], R["ratio_deployed_ladder"])
         if (int(f_), float(m_)) in key]
    print("wiring: ratio at the true direction against s3_window_check_800.npz, %d runs, max |diff| %.2e" % (len(d), max(d)))

idx = np.arange(len(P))
for mxx in (0.25, 1.0, 4.0):
    print("\n== eps x%g" % mxx)
    print("   %-9s %4s  %-12s %-22s %-18s %-16s %-16s %-14s"
          % ("profile", "n", "ratio ladder", "stops lower when noisy", "misfit m / tau_min", "rho^2/(tm^2+m^2)", "ratio at clean stop", "ratio at deg 20"))
    for nm in sorted(set(P)):
        k = (R["eps_mult"] == mxx) & (P == nm)
        ii = idx[k]
        tm = R["tau_min"][k]
        sn, sc_ = R["stop_noisy"][k], R["stop_clean"][k]
        m = R["r_clean"][ii, sn]
        rho = R["r_noisy"][ii, sn]
        print("   %-9s %4d  %-12.2f %-22s %-18.2f %-16.2f %-16.2f %-14.2f"
              % (nm, k.sum(), np.median(rho / tm), "%.0f%%" % (100 * np.mean(sn < sc_)), np.median(m / tm),
                 np.median(rho ** 2 / (tm ** 2 + m ** 2)), np.median(R["r_noisy"][ii, sc_] / tm), np.median(R["r_noisy"][ii, ND - 1] / tm)))
    k = R["eps_mult"] == mxx
    for nm in ("softplus", "exp"):
        kk = k & (P == nm)
        print("   %-9s stopping degree noisy: %s | noiseless: %s"
              % (nm, dict(zip(*np.unique(np.array(DEGREES)[R["stop_noisy"][kk]], return_counts=True))),
                 dict(zip(*np.unique(np.array(DEGREES)[R["stop_clean"][kk]], return_counts=True)))))

out = os.path.join(HERE, "data", "s3_degree_check_%d.npz" % N_NULL)
np.savez(out, degrees=np.array(DEGREES), profile=P, **R)
print("\nsaved", out)
