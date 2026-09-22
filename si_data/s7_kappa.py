# -*- coding: utf-8 -*-
"""Supplementary Table 4 -- weight heterogeneity and the quadrature bound of Supplementary Note 3.

The null simulation of a run estimates one phi_i per anchor, phi_i = mean_r exp(2i delta_ir),
and sigma_noise^2 = -1/2 ln |mean_i phi_i|.  With
    eps_i   = phi_i / mean_i(phi_i) - 1          complex, sums to zero over the anchors
    eps_max = max_i |eps_i|,   eps_rms = sqrt(mean_i |eps_i|^2)
    kappa  <= 2 eps_rms sigma exp(2 sigma^2)     sigma = the run's sigma_true_hat, in radians
the relative error of sigma_true_hat is bounded by eps exp(2 sigma^2) / (sigma (1 - kappa)),
leading order eps / (2 sigma).

Runs: the eight benchmarks in the realisation of benchmarks/data/fig2_method_data.npz (the
recomputed sigma_noise is asserted equal to the stored one), and the first N_SYN fields of
data/s5_tau1_1200.npz at eps x 1 and x 4 (same seeds and draws as s5_tau1_sweep.py).

Env: N_SYN (100).  Prints only.  About 7 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework import core as F                                   # noqa: E402
from framework.benchmarks import UNITS                            # noqa: E402
from framework.core import M_ANCH, N_MERGE, N_MC, N_FIELDSIM, RSEED, build, plan_uniform, merged_run, null_sim  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, draw_lam, calibrate_eps, S5_SEED0 as SEED0  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_SYN = int(os.environ.get("N_SYN", 100))
BENCH_DATA = os.path.join(HERE, "..", "benchmarks", "data", "fig2_method_data.npz")


def bounds(phi, st_deg):
    """eps_max, eps_rms and, for each, [kappa, rigorous relative-error bound, leading order]"""
    eps = phi / phi.mean() - 1.0
    e_max = float(np.max(np.abs(eps)))
    e_rms = float(np.sqrt(np.mean(np.abs(eps) ** 2)))
    s = math.radians(st_deg)
    g = math.exp(2 * s * s)
    out = []
    for e in (e_max, e_rms):
        kap = 2 * e * s * g
        rel = e * g / (s * (1 - kap)) if s > 0 and kap < 1 else float("nan")
        lead = e / (2 * s) if s > 0 else float("nan")
        out += [kap, rel, lead]
    return e_max, e_rms, out


print("=== eight benchmarks: eps_max / eps_rms and the bounds (kappa, rigorous relative-error bound, leading order)")
D = np.load(BENCH_DATA, allow_pickle=True)
print("%-22s %7s %7s %8s | %7s %7s %7s %7s | %7s %7s %7s %7s"
      % ("regime", "st_hat", "sn", "min|phi|", "eps_max", "kappa", "rel", "lead", "eps_rms", "kappa", "rel", "lead"))
t0 = time.time()
for ui, (name, key, sub, red) in enumerate(UNITS):
    seed = RSEED + ui
    eps = F.calibrate_eps(key, sub, seed)
    CX, CY, HX, HY, FC, F0 = build(key, sub, M_ANCH, seed)
    SFC = FC - F0[:, None]
    S_fr = plan_uniform(M_ANCH)
    rng2 = np.random.default_rng(seed * 13)                         # the realisation of fig2_method_data.npz
    prof, pool_r, _yn, _E = merged_run(SFC, F0, eps, rng2)
    sn, _q95, phi, _sa_null = null_sim(prof, S_fr, HX, HY, eps, np.random.default_rng(rng2.integers(1 << 30)),
                                       N_MC, N_FIELDSIM, pool=pool_r, nrep=N_MERGE, return_null=True)
    p = "u%d_" % ui
    assert str(D[p + "name"]) == name
    assert abs(sn - float(D[p + "sn"])) < 1e-9, (name, sn, float(D[p + "sn"]))
    st = float(D[p + "st"])
    e_max, e_rms, b = bounds(phi, st)
    print("%-22s %7.2f %7.2f %8.4f | %7.4f %7.4f %7.3f %7.3f | %7.4f %7.4f %7.3f %7.3f"
          % (name[:22], st, sn, np.abs(phi).min(), e_max, b[0], b[1], b[2], e_rms, b[3], b[4], b[5]), flush=True)
print("  %.0f s" % (time.time() - t0))

print("\n=== synthetic fields of Supplementary Fig. 5, first %d fields x two noise levels" % N_SYN)
S5 = np.load(os.path.join(HERE, "data", "s5_tau1_1200.npz"), allow_pickle=True)
A = S5["rows"]
c = {str(k): i for i, k in enumerate(S5["cols"])}
stored = {(int(r[c["field"]]), float(r[c["eps_mult"]])): r for r in A}
rows = {1.0: [], 4.0: []}
t0 = time.time()
for j in range(N_SYN):
    seed = SEED0 + j
    nm, f, sc, a_, b_ = draw_ridge(seed)
    lam = draw_lam(seed)
    eps0 = calibrate_eps(mk(sc, f, a_, b_, 0.0), seed)
    CX, CY, HX, HY, FC, F0 = build_field(mk(sc, f, a_, b_, lam), M_ANCH, seed)
    SFC = FC - F0[:, None]
    S = plan_uniform(M_ANCH)
    for mxx in (1.0, 4.0):
        rg = np.random.default_rng(seed * 19 + int(mxx * 1e6))      # the noise draw of s5_tau1_sweep.py
        prof_, pool, _yn, _E = merged_run(SFC, F0, mxx * eps0, rg)
        sn, _q95, phi, _sa_null = null_sim(prof_, S, HX, HY, mxx * eps0, np.random.default_rng(rg.integers(1 << 30)),
                                           N_MC, N_FIELDSIM, pool=pool, nrep=N_MERGE, return_null=True)
        row = stored[(j, mxx)]
        assert abs(sn - row[c["sigma_noise"]]) < 1e-9, (j, mxx)
        st = row[c["sigma_true_hat"]]
        e_max, e_rms, b = bounds(phi, st)
        #                0    1                     2   3   4      5      6     7     8     9     10    11    12
        rows[mxx].append((lam, row[c["sigma_true"]], st, sn, e_max, e_rms, b[0], b[3], b[1], b[4], b[2], b[5], np.abs(phi).min()))
print("  %d runs in %.0f s; recomputed sigma_noise equals the stored value on every run" % (2 * N_SYN, time.time() - t0))

for mxx in (1.0, 4.0):
    R = np.array(rows[mxx])
    tilt = R[:, 0] > 0
    print("  eps x%g (n %d): eps_max median %.3f p90 %.3f max %.3f | eps_rms median %.4f p90 %.4f max %.4f"
          " | kappa(rms) max %.4f | min|phi| median %.3f min %.3f"
          % (mxx, len(R), np.median(R[:, 4]), np.percentile(R[:, 4], 90), R[:, 4].max(),
             np.median(R[:, 5]), np.percentile(R[:, 5], 90), R[:, 5].max(), R[:, 7].max(), np.median(R[:, 12]), R[:, 12].min()))
    for lo, hi in ((2, 4), (4, 8), (8, 90)):
        k = tilt & (R[:, 2] >= lo) & (R[:, 2] < hi)
        if k.sum():
            print("     st_hat in [%d, %d) deg: n %2d | rel-error bound (eps_max) median %.3f max %.3f"
                  " | (eps_rms) median %.3f max %.3f | leading order (eps_rms) median %.3f"
                  % (lo, hi, k.sum(), np.median(R[k, 8]), R[k, 8].max(), np.median(R[k, 9]), R[k, 9].max(), np.median(R[k, 11])))
