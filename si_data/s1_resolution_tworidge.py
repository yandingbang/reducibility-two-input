# -*- coding: utf-8 -*-
"""Supplementary Fig. 1 data, second family -- the two-ridge fields.

Same as s1_resolution_linear.py with the transverse term replaced by a second ridge
(framework.synthetic.mk2): the main ridge of field j is that of the first family (same
seed 51300000 + j), the second ridge comes from an independent stream (draw_second).  The
noise level is calibrated on the lam = 0 field, which is the same field in both families,
and the observation rng is seeded in the same way, so at a lam value common to both
designs the two families receive the same noise draw.

Env: N_FIELD (100), EPSX ("1,4"), LAMS ("0,0.002,0.005,0.01,0.02,0.05,0.10,0.30").
Output: data/s1_tworidge_<multiple>.npz.  About 15 min per file.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.core import M_ANCH, N_MERGE, N_MC, N_FIELDSIM, plan_uniform, merged_run, null_sim, read_interp, axial  # noqa: E402
from framework.synthetic import build_field, mk2, draw_ridge, draw_second, calibrate_eps, sigma_true  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NF = int(os.environ.get('N_FIELD', 100))
EPSX_R = [float(v) for v in os.environ.get('EPSX', '1,4').split(',')]
LAMS_R = [float(v) for v in os.environ.get('LAMS', '0,0.002,0.005,0.01,0.02,0.05,0.10,0.30').split(',')]
SEED0 = 51300000

print('resolution, two ridges: N_FIELD %d, eps x %s, %d lam, M %d, N_MC %d, N_F %d, n %d'
      % (NF, EPSX_R, len(LAMS_R), M_ANCH, N_MC, N_FIELDSIM, N_MERGE), flush=True)
T0 = time.time()
for mxx in EPSX_R:
    rows = []
    for lam in LAMS_R:
        for j in range(NF):
            seed = SEED0 + j
            nm, f, sc, a_, b_ = draw_ridge(seed)
            nm2, g, scg, c_, d_ = draw_second(seed, a_, b_)
            eps0 = calibrate_eps(mk2(sc, f, a_, b_, 0.0, scg, g, c_, d_), seed)
            if not np.isfinite(eps0):
                continue
            eps = mxx * eps0
            resp = mk2(sc, f, a_, b_, lam, scg, g, c_, d_)
            CX, CY, HX, HY, FC, F0 = build_field(resp, M_ANCH, seed)
            if not np.all(np.isfinite(FC)):
                continue
            SFC = FC - F0[:, None]
            S = plan_uniform(M_ANCH)
            s_true, _ = sigma_true(resp, CX, CY)
            rg = np.random.default_rng(seed * 19 + int(mxx * 1e6) + int(lam * 1e7))
            prof, pool, _yn, _E = merged_run(SFC, F0, eps, rg)
            sn, q95 = null_sim(prof, S, HX, HY, eps, np.random.default_rng(rg.integers(1 << 30)),
                               N_MC, N_FIELDSIM, pool=pool, nrep=N_MERGE)
            sa = axial(read_interp(prof, S, HX, HY))[1]
            st = F.sigma_true_hat(sa, sn)
            rows.append((mxx, lam, j, s_true, sn, q95, st, float(st > q95)))
        print('  eps x%-5g lam %-6g done, %.0f s' % (mxx, lam, time.time() - T0), flush=True)
    A = np.array(rows, float)
    out = os.path.join(HERE, 'data', 's1_tworidge_%g.npz' % mxx)
    np.savez_compressed(out, rows=A)
    print('saved %s  (%d rows)' % (out, len(A)), flush=True)
