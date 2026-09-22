# -*- coding: utf-8 -*-
"""Supplementary Fig. 5 data -- what the declared tolerance tau_1 trades: errors or abstentions.

Fields: single ridge + linear term, seeds 63300000 + j.  lam is drawn per field from its own
stream (7 s + 3): 0 with probability P_EXACT, otherwise log-uniform on [LAM_LO, LAM_HI], so
that sigma_true covers the tau_1 range continuously.  Each field is run at eps x 1 and x 4 of
its own calibration, one complete stage one each (observation rng 19 s + 1e6 multiple).
Nothing in a run depends on tau_1: the sweep (s5_tau1_summary.py, figures/si/plot_s5_tau1.py)
re-scores the stored (sigma_true, q95, sigma_true_hat) at every tau_1.

Env: N_FIELD (1200).  Output: data/s5_tau1_<N_FIELD>.npz.  About 1 h.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, N_MC, N_FIELDSIM, plan_uniform, merged_run, null_sim, read_interp, axial  # noqa: E402
from framework.synthetic import (build_field, mk, draw_ridge, draw_lam, calibrate_eps, sigma_true,  # noqa: E402
                                 S5_SEED0 as SEED0, S5_P_EXACT as P_EXACT, S5_LAM_LO as LAM_LO, S5_LAM_HI as LAM_HI)

HERE = os.path.dirname(os.path.abspath(__file__))
N_FIELD = int(os.environ.get("N_FIELD", 1200))
LEVELS = (1.0, 4.0)
COLS = ("field", "lam", "eps_mult", "sigma_true", "sigma_noise", "q95", "sigma_alpha", "sigma_true_hat")

print("tau_1 sweep data: N_FIELD %d x levels %s, M %d, N_MERGE %d, N_MC %d, N_FIELDSIM %d"
      % (N_FIELD, LEVELS, M_ANCH, N_MERGE, N_MC, N_FIELDSIM), flush=True)
rows, prof = [], []
t0 = time.time()
for j in range(N_FIELD):
    seed = SEED0 + j
    nm, f, sc, a_, b_ = draw_ridge(seed)
    lam = draw_lam(seed)
    eps0 = calibrate_eps(mk(sc, f, a_, b_, 0.0), seed)
    if not np.isfinite(eps0):
        continue
    resp = mk(sc, f, a_, b_, lam)
    CX, CY, HX, HY, FC, F0 = build_field(resp, M_ANCH, seed)
    if not np.all(np.isfinite(FC)):
        continue
    SFC = FC - F0[:, None]
    S = plan_uniform(M_ANCH)
    s_true, _ = sigma_true(resp, CX, CY)
    for mxx in LEVELS:
        eps = mxx * eps0
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))
        prof_, pool, _yn, _E = merged_run(SFC, F0, eps, rg)
        sn, q95 = null_sim(prof_, S, HX, HY, eps, np.random.default_rng(rg.integers(1 << 30)),
                           N_MC, N_FIELDSIM, pool=pool, nrep=N_MERGE)
        sa = float(axial(read_interp(prof_, S, HX, HY))[1])
        rows.append((j, lam, mxx, s_true, sn, q95, sa, F.sigma_true_hat(sa, sn)))
        prof.append(nm)
    if (j + 1) in (1, 5) or (j + 1) % 100 == 0:
        print("  field %4d  runs %5d  %.0f s" % (j + 1, len(rows), time.time() - t0), flush=True)

A = np.array(rows, float)
print("\n%d runs in %.0f s" % (len(A), time.time() - t0))
st1 = A[A[:, 2] == 1.0, 3]
print("sigma_true over fields: exactly reducible %d, quartiles of the rest %s deg"
      % (np.sum(st1 == 0), " / ".join("%.2f" % v for v in np.percentile(st1[st1 > 0], [25, 50, 75])) if np.any(st1 > 0) else "-"))
out = os.path.join(HERE, "data", "s5_tau1_%d.npz" % N_FIELD)
np.savez(out, rows=A, cols=np.array(COLS), profile=np.array(prof), seed0=SEED0, p_exact=P_EXACT,
         lam_lo=LAM_LO, lam_hi=LAM_HI, n_merge=N_MERGE, n_mc=N_MC, n_fieldsim=N_FIELDSIM, m_anch=M_ANCH)
print("saved", out)
