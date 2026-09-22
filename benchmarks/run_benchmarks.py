# -*- coding: utf-8 -*-
"""The framework end to end on the eight benchmarks, N_REP noise realisations each.

Every realisation is one complete run of the protocol (its own repeats, residual pool and
null simulation) followed by both stages and the joint verdict.  Per regime the script
reports the medians of the statistics and the mode of the verdicts over the realisations,
and how often the mode occurred.  The first realisation of each regime (rng seed 13 s, s =
RSEED + unit index) is the one shown in the main text; fig2_data.py regenerates it alone.

Environment: N_REP (8), TAU1 (4), N_MERGE (6), N_MC (1200), N_FIELDSIM (6400).
Output: data/framework_pipeline_v4.npz.  Runtime about 4 min at N_REP = 8 (8 x 8 null
simulations); the realisations are drawn from one stream per regime, so a larger N_REP
extends the same sequence.
"""
import math
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.benchmarks import UNITS                            # noqa: E402
from framework.core import (M_ANCH, NQ, FRAC_MIN, SIG_REL, N_MERGE, N_MC, N_FIELDSIM, TAU, TAU1,  # noqa: E402
                            RSEED, build, plan_uniform, merged_run, read_interp, axial, null_sim,
                            rho_poly, rho_np, window_rho, v1, v2, joint)

HERE = os.path.dirname(os.path.abspath(__file__))
N_REP = int(os.environ.get('N_REP', 8))
OUT_FILE = os.path.join(HERE, 'data', os.environ.get('OUTNPZ', 'framework_pipeline_v4.npz'))

print('=' * 100)
print('FRAMEWORK, END TO END   M=%d  K=%d  n=%d  margin %.2f  probe as large as fits  N_MC=%d  N_F=%d'
      % (M_ANCH, NQ, N_MERGE, FRAC_MIN, N_MC, N_FIELDSIM))
print('eps = %.0f%% of the median amplitude at the smallest probe, calibrated once;  tau_1 = %.0f deg  tau = %.0f%%'
      % (100 * SIG_REL, TAU1, 100 * TAU))
print('=' * 100)

rows = []
t_start = time.time()
for ui, (name, key, sub, red) in enumerate(UNITS):
    seed = RSEED + ui
    eps = F.calibrate_eps(key, sub, seed)
    CX, CY, HX, HY, FC, F0 = build(key, sub, M_ANCH, seed)
    SFC = FC - F0[:, None]
    S_fr = plan_uniform(M_ANCH)

    sa_l, st_l, rp_l, rn_l, tm_l, rt_l, sn_l, q95_l = ([] for _ in range(8))
    rng2 = np.random.default_rng(seed * 13)
    for _ in range(N_REP):
        prof, pool_r, yn, _E = merged_run(SFC, F0, eps, rng2)
        s_noise, q95 = null_sim(prof, S_fr, HX, HY, eps,
                                np.random.default_rng(rng2.integers(1 << 30)),
                                N_MC, N_FIELDSIM, pool=pool_r, nrep=N_MERGE)
        ab, sa = axial(read_interp(prof, S_fr, HX, HY))
        rp, a_star = window_rho(CX, CY, yn, ab, sa, rho_poly)
        rn, _ = window_rho(CX, CY, yn, ab, sa, rho_np, angle=a_star)
        tm = F.tau_min(pool_r, yn)
        sn_l.append(s_noise); q95_l.append(q95); sa_l.append(sa)
        st_l.append(F.sigma_true_hat(sa, s_noise))
        rp_l.append(rp); rn_l.append(rn); tm_l.append(tm)
        rt_l.append(F.rho_true_hat(rp, tm))

    c1l = [v1(s, TAU1, q) for s, q in zip(st_l, q95_l)]
    c2l = [v2(r, t) for r, t in zip(rt_l, tm_l)]
    jl = [joint(a, b) for a, b in zip(c1l, c2l)]
    s1, s2, jt = (max(set(v), key=v.count) for v in (c1l, c2l, jl))
    jfrac = jl.count(jt) / len(jl)
    ok = (jt != 'UND') and ((jt == 'RED') == bool(red))
    med = lambda v: float(np.median(v))
    rows.append(dict(name=name, red=red, eps=eps, s_y=float(np.std(F0, ddof=1)),
                     s_noise=med(sn_l), q95=med(q95_l), sa=med(sa_l), sa_sd=float(np.std(sa_l, ddof=1)),
                     st=med(st_l), rp=med(rp_l), rn=med(rn_l), tm=med(tm_l), rt=med(rt_l),
                     s1=s1, s2=s2, jt=jt, jfrac=jfrac, ok=ok))
    r = rows[-1]
    print('\n%s   [truth: %s]   %.0f s' % (name, 'reducible' if red else 'NOT reducible', time.time() - t_start))
    print('   eps = %.4g   s_y = %.4g' % (eps, r['s_y']))
    print('   STAGE 1   sigma_alpha %6.2f +/-%.2f   sigma_noise %5.2f   q95 %5.2f   sigma_true_hat %6.2f'
          '  vs max(tau_1, q95) = %5.2f   -> %s' % (r['sa'], r['sa_sd'], r['s_noise'], r['q95'], r['st'],
                                                    max(TAU1, r['q95']), s1))
    print('   STAGE 2   rho %.4f   rho_np %.4f   tau_min %.4f   rho_true_hat %.4f  vs max(tau, tau_min) = %.4f'
          '   -> %s' % (r['rp'], r['rn'], r['tm'], r['rt'], max(TAU, r['tm']), s2))
    print('   OUTCOME   %s  (%.0f%% of %d realisations)   %s'
          % (jt, 100 * jfrac, N_REP, 'correct' if ok else ('UNRESOLVED' if jt == 'UND' else '<-- DIFFERS FROM TRUTH')))

print('\n' + '=' * 100)
print('%-24s %7s %7s %9s %8s %8s %9s %8s %10s' % ('regime', 'sig_a', 'q95', 'sig_true', 'rho', 'rho_np', 'tau_min', 'verdict', 'truth'))
for r in rows:
    print('%-24s %7.2f %7.2f %9.2f %8.4f %8.4f %9.4f %8s %10s'
          % (r['name'], r['sa'], r['q95'], r['st'], r['rp'], r['rn'], r['tm'], r['jt'], 'reduce' if r['red'] else 'NOT red'))
print('  joint verdicts matching the truth label: %d of %d; ' % (sum(r['ok'] for r in rows), len(rows))
      + ', '.join('%s=%d' % kv for kv in sorted(Counter(r['jt'] for r in rows).items())))

RED = np.array([r['red'] for r in rows])
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez(OUT_FILE,
         units=np.array([r['name'] for r in rows]), reducible=RED,
         eps=np.array([r['eps'] for r in rows]), s_y=np.array([r['s_y'] for r in rows]),
         sigma_alpha=np.array([r['sa'] for r in rows]), sigma_alpha_sd=np.array([r['sa_sd'] for r in rows]),
         floor=np.array([r['s_noise'] for r in rows]), q95=np.array([r['q95'] for r in rows]),
         sigma_true=np.array([r['st'] for r in rows]),
         rho_poly=np.array([r['rp'] for r in rows]), rho_np=np.array([r['rn'] for r in rows]),
         tau_min=np.array([r['tm'] for r in rows]), rho_true=np.array([r['rt'] for r in rows]),
         stage1=np.array([r['s1'] for r in rows]), stage2=np.array([r['s2'] for r in rows]),
         joint=np.array([r['jt'] for r in rows]), joint_frac=np.array([r['jfrac'] for r in rows]),
         tau1=TAU1, tau=TAU, M=M_ANCH, frac_min=FRAC_MIN, n_rep=N_REP)
print('saved', OUT_FILE)
