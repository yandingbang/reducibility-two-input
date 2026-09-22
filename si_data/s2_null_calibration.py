# -*- coding: utf-8 -*-
"""Supplementary Fig. 2a data -- is q95 a 95th percentile on one exactly reducible field?

Field: the anchors and probes of the low-Re pipe benchmark (M_ANCH anchors, seed RSEED) with
the response replaced by y = exp(0.7 (0.8 u - 0.6 v)), so sigma_true = 0 exactly; eps is
SIG_REL times the median amplitude over the deployed probes.  N_RUN independent noise
realisations from one stream (seed 20260910): each draws the merged run, then an integer that
seeds its null simulation.  Kept per run: sigma_noise, q95, sigma_alpha, sigma_true_hat,
exceed (sigma_true_hat > q95) and p, the share of the run's own null draws of sigma_true_hat
that reach the observed one.

Env: N_RUN (600).  Output: data/s2_null_calib_<N_RUN>.npz.  About 15 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import (M_ANCH, SIG_REL, N_MERGE, N_MC, N_FIELDSIM, RSEED, SIZE, _t,  # noqa: E402
                            build, plan_uniform, merged_run, null_sim, read_interp, axial)

HERE = os.path.dirname(os.path.abspath(__file__))
N_RUN = int(os.environ.get("N_RUN", 600))
SEED = 20260910


def reducible_field(CX, CY, HX, HY, a=0.8, b=-0.6, Fn=lambda z: np.exp(0.7 * z)):
    """y = F(a u + b v): sigma_true = 0 exactly."""
    F0 = Fn(a * CX + b * CY)
    FC = np.empty((CX.size, SIZE))
    for k in range(SIZE):
        t = _t[k]
        FC[:, k] = Fn(a * (CX + HX * math.cos(t)) + b * (CY + HY * math.sin(t)))
    return FC, F0


CX, CY, HX, HY, _FC, _F0 = build("Pipe low-Re", None, M_ANCH, RSEED)
FC, F0 = reducible_field(CX, CY, HX, HY)
SFC = FC - F0[:, None]
eps = SIG_REL * float(np.median(np.abs(SFC).max(1)))
S = plan_uniform(M_ANCH)
print("exactly reducible field: M %d, eps %.4g, N_MERGE %d, N_MC %d, N_FIELDSIM %d, runs %d"
      % (M_ANCH, eps, N_MERGE, N_MC, N_FIELDSIM, N_RUN))
print("noise-free sigma_alpha = %.6f deg (should be 0)" % axial(read_interp(SFC, S, HX, HY))[1], flush=True)

rg = np.random.default_rng(SEED)
rec = {k: [] for k in ("sn", "q95", "sa", "st", "exceed", "p")}
t0 = time.time()
for r in range(N_RUN):
    prof, pool, _yn, _E = merged_run(SFC, F0, eps, rg)
    s = int(rg.integers(1 << 30))
    sn, q95, _phi, sa_null = null_sim(prof, S, HX, HY, eps, np.random.default_rng(s),
                                      N_MC, N_FIELDSIM, pool=pool, nrep=N_MERGE, return_null=True)
    sa = axial(read_interp(prof, S, HX, HY))[1]
    st = F.sigma_true_hat(sa, sn)
    st_null = np.sqrt(np.maximum(sa_null ** 2 - sn ** 2, 0.0))
    for k, v in (("sn", sn), ("q95", q95), ("sa", sa), ("st", st),
                 ("exceed", float(st > q95)), ("p", float(np.mean(st_null >= st)))):
        rec[k].append(v)
    if r + 1 == 1 or (r + 1) % 50 == 0:
        print("  run %4d  %.1f s" % (r + 1, time.time() - t0), flush=True)

A = {k: np.array(v) for k, v in rec.items()}
n = N_RUN
rate = A["exceed"].mean()
print("\nabove own q95: %d of %d = %.2f%%  (nominal 5%%, binomial s.e. %.2f%%)"
      % (int(A["exceed"].sum()), n, 100 * rate, 100 * math.sqrt(0.05 * 0.95 / n)))
for a in (0.01, 0.05, 0.10, 0.20):
    print("  p <= %.2f: %.2f%%" % (a, 100 * float(np.mean(A["p"] <= a))))
print("median sigma_noise %.3f, q95 %.3f, sigma_true_hat %.3f deg; %.1f s per run"
      % (np.median(A["sn"]), np.median(A["q95"]), np.median(A["st"]), (time.time() - t0) / n))
out = os.path.join(HERE, "data", "s2_null_calib_%d.npz" % n)
np.savez(out, eps=eps, seed=SEED, n_merge=N_MERGE, n_mc=N_MC, n_fieldsim=N_FIELDSIM, m_anch=M_ANCH, **A)
print("saved", out)
