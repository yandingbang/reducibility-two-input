# -*- coding: utf-8 -*-
"""Supplementary Fig. 2b data -- is q95 calibrated across many exactly reducible fields?

Fields: single ridge, lam = 0 (exactly reducible, sigma_true = 0), seeds 61300000 + j.  Each
field is run at eps x 1 and x 4 of its own calibration, one complete stage one each
(observation rng 19 s + 1e6 multiple; the null simulation seeded with an integer drawn from it).
Kept per run: field index, noise multiple, eps, sigma_noise, q95, sigma_alpha, sigma_true_hat,
exceed (sigma_true_hat > q95), p (share of the run's own null draws reaching sigma_true_hat)
and floor, the field's noise-free sigma_alpha (all of it read-out error, since sigma_true = 0).

Env: N_FIELD (800).  Output: data/s2_null_fields_<N_FIELD>.npz.  About 40 min.
"""
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, N_MC, N_FIELDSIM, plan_uniform, merged_run, null_sim, read_interp, axial  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, calibrate_eps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_FIELD = int(os.environ.get("N_FIELD", 800))
LEVELS = (1.0, 4.0)
SEED0 = 61300000
COLS = ("field", "eps_mult", "eps", "sn", "q95", "sa", "st", "exceed", "p", "floor")


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi


print("many exactly reducible fields: N_FIELD %d x levels %s, M %d, N_MERGE %d, N_MC %d, N_FIELDSIM %d"
      % (N_FIELD, LEVELS, M_ANCH, N_MERGE, N_MC, N_FIELDSIM), flush=True)
rows, prof_names = [], []
t0 = time.time()
for j in range(N_FIELD):
    seed = SEED0 + j
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
    floor = float(axial(read_interp(SFC, S, HX, HY))[1])
    for mxx in LEVELS:
        eps = mxx * eps0
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))
        prof, pool, _yn, _E = merged_run(SFC, F0, eps, rg, S)
        s = int(rg.integers(1 << 30))
        sn, q95, _phi, sa_null = null_sim(prof, S, HX, HY, eps, np.random.default_rng(s),
                                          N_MC, N_FIELDSIM, pool=pool, nrep=N_MERGE, return_null=True)
        sa = float(axial(read_interp(prof, S, HX, HY))[1])
        st = F.sigma_true_hat(sa, sn)
        st_null = np.sqrt(np.maximum(sa_null ** 2 - sn ** 2, 0.0))
        rows.append((j, mxx, eps, sn, q95, sa, st, float(st > q95), float(np.mean(st_null >= st)), floor))
        prof_names.append(nm)
    if (j + 1) in (1, 5) or (j + 1) % 50 == 0:
        print("  field %4d  runs %4d  %.0f s" % (j + 1, len(rows), time.time() - t0), flush=True)

A = np.array(rows, float)
print("\n%d runs in %.0f s, %.2f s per run" % (len(A), time.time() - t0, (time.time() - t0) / max(len(A), 1)))
for mxx in LEVELS:
    B = A[A[:, 1] == mxx]
    n, k = len(B), int(B[:, 7].sum())
    lo, hi = cp(k, n)
    ratio = B[:, 4] / np.maximum(B[:, 9], 1e-12)
    print("eps x%g: above own q95 %d of %d = %.2f%% (95%% CI %.2f-%.2f%%); p <= 1/5/10/20%%: %s"
          % (mxx, k, n, 100.0 * k / n, lo, hi,
             " / ".join("%.1f%%" % (100 * np.mean(B[:, 8] <= a)) for a in (0.01, 0.05, 0.10, 0.20))))
    print("   median q95 %.3f, median read-out floor %.3f, q95/floor quartiles %s"
          % (np.median(B[:, 4]), np.median(B[:, 9]), " / ".join("%.1f" % v for v in np.percentile(ratio, [25, 50, 75]))))
out = os.path.join(HERE, "data", "s2_null_fields_%d.npz" % N_FIELD)
np.savez(out, rows=A, cols=np.array(COLS), profile=np.array(prof_names), seed0=SEED0,
         n_merge=N_MERGE, n_mc=N_MC, n_fieldsim=N_FIELDSIM, m_anch=M_ANCH)
print("saved", out)
