# -*- coding: utf-8 -*-
"""One realisation of the protocol on each of the eight benchmarks: the data of Figs. 1b, 2-5
and Table 1 of the main text.

For every regime (framework.benchmarks.UNITS, seed RSEED + unit index):
    the noise level eps, calibrated at 200 preliminary anchors (seed + 7777);
    the 40 anchors and probes, the query plan, the merged run (N_MERGE repeats, seed 13 s);
    stage one: the null simulation (sigma_noise, q95 and the null distribution itself),
               the 40 directions, alpha_bar, sigma_alpha, sigma_true_hat and the verdict.
Stage two is recomputed from the saved anchors and averaged responses by figures/common.py
(window search, rho, tau_min, rho_true_hat), so that every figure uses the same functions.

The anchor highlighted in Fig. 2b is chosen here: among anchors whose profile at the queried
angles has exactly one sign change, the one at the 75th percentile of the amplitude-to-noise
ratio R / eps.

Output: data/fig2_method_data.npz (about 2.5 MB).  Runtime about 4 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.benchmarks import CASES, UNITS                     # noqa: E402
from framework.core import (M_ANCH, NQ, N_MERGE, N_MC, N_FIELDSIM, TAU, TAU1, RSEED, SIZE, _t,  # noqa: E402
                            build, plan_uniform, merged_run, interp_cross, read_interp, axial, v1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, 'data', 'fig2_method_data.npz')


def pick_anchor(prof, idx_q, eps):
    """The anchor shown in Fig. 2b, and the per-anchor R / eps and sign-change counts."""
    sq = prof[:, idx_q]
    R = np.ptp(sq, axis=1)
    ns = np.array([int(np.sum(np.sign(r[:-1]) * np.sign(r[1:]) < 0)) for r in sq])
    ok = np.where(ns == 1)[0]
    ok = ok if ok.size else np.arange(len(R))
    order = ok[np.argsort(R[ok] / eps)]
    return int(order[int(0.75 * (len(order) - 1))]), R / eps, ns


print('protocol  M=%d  K=%d  n=%d  N_MC=%d  N_F=%d  tau_1=%.0f deg  tau=%.2f'
      % (M_ANCH, NQ, N_MERGE, N_MC, N_FIELDSIM, TAU1, TAU))
OUT = {}
t0 = time.time()
for ui, (name, key, sub, red) in enumerate(UNITS):
    seed = RSEED + ui
    eps = F.calibrate_eps(key, sub, seed)
    CX, CY, HX, HY, FC, F0 = build(key, sub, M_ANCH, seed)
    SFC = FC - F0[:, None]                          # noise-free signed profile, SIZE angles
    S_fr = plan_uniform(M_ANCH)
    Sa0 = np.array(sorted(S_fr[0]))                 # the NQ queried angles, same at every anchor

    rng2 = np.random.default_rng(seed * 13)
    prof, pool_r, yn, E = merged_run(SFC, F0, eps, rng2, S_fr)
    sn, q95, _phi, sa_null = F.null_sim(
        prof, S_fr, HX, HY, eps, np.random.default_rng(rng2.integers(1 << 30)),
        N_MC, N_FIELDSIM, pool=pool_r, nrep=N_MERGE, return_null=True)
    st_null = np.sqrt(np.maximum(sa_null ** 2 - sn ** 2, 0.0))
    ang = read_interp(prof, S_fr, HX, HY)           # the 40 directions (radians, mod pi)
    ab, sa = axial(ang)
    st = F.sigma_true_hat(sa, sn)
    verdict = v1(st, TAU1, q95)

    prof_reps = np.full((N_MERGE,) + SFC.shape, np.nan)         # each repeat's profile (n, 40, SIZE), nan off the plan
    prof_reps[:, :, Sa0] = SFC[None, :, Sa0] + E[:, :, :-1] - E[:, :, -1:]
    t0_cross = np.array([interp_cross(prof[i, Sa0], _t[Sa0]) for i in range(M_ANCH)])
    pick, r_over_eps, n_signchg = pick_anchor(prof, Sa0, eps)

    box = CASES[key]['boxes'][0 if sub is None else sub]
    p = 'u%d_' % ui
    OUT[p + 'name'] = name
    OUT[p + 'reducible'] = red
    OUT[p + 'box'] = np.array(box, float)           # x0 x1 y0 y1
    OUT[p + 'CX'], OUT[p + 'CY'] = CX, CY
    OUT[p + 'HX'], OUT[p + 'HY'] = HX, HY
    OUT[p + 't_all'] = _t
    OUT[p + 't_q'] = _t[Sa0]
    OUT[p + 'idx_q'] = Sa0
    OUT[p + 'prof_true'] = SFC
    OUT[p + 'prof_mean'] = prof
    OUT[p + 'prof_reps'] = prof_reps
    OUT[p + 'cross_t'] = t0_cross
    OUT[p + 'ang'] = ang
    OUT[p + 'pick'] = pick
    OUT[p + 'R_over_eps'] = r_over_eps
    OUT[p + 'n_signchg'] = n_signchg
    OUT[p + 'eps'] = eps
    OUT[p + 'yn'] = yn
    OUT[p + 's_y'] = float(np.std(yn, ddof=1))
    OUT[p + 'eps_pool'] = float(np.std(pool_r, ddof=1))
    OUT[p + 'sa'] = sa
    OUT[p + 'abar'] = ab
    OUT[p + 'sn'] = sn
    OUT[p + 'q95'] = q95
    OUT[p + 'st'] = st
    OUT[p + 'verdict'] = verdict
    OUT[p + 'st_null'] = st_null
    OUT[p + 'sa_null'] = sa_null
    print('  %-24s sigma_alpha %6.3f  sigma_noise %6.3f  q95 %6.3f  sigma_true_hat %7.3f  -> %-10s  %.0f s'
          % (name, sa, sn, q95, st, verdict, time.time() - t0), flush=True)

OUT['TAU1'] = TAU1
OUT['TAU'] = TAU
OUT['M_ANCH'] = M_ANCH
OUT['N_MERGE'] = N_MERGE
OUT['N_MC'] = N_MC
OUT['N_FIELDSIM'] = N_FIELDSIM
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez_compressed(OUT_FILE, **OUT)
print('saved %s  (%.1f MB, %.0f s)' % (OUT_FILE, os.path.getsize(OUT_FILE) / 1e6, time.time() - t0))
