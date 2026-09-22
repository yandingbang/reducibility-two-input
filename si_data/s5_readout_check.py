# -*- coding: utf-8 -*-
"""Supplementary Fig. 5 check -- why a field above tau_1 can be called reducible.

For a field with sigma_true > tau_1 the estimate sigma_true_hat has to fall below the truth;
three steps can do that: the noise-free read-out (sigma_alpha_clean differs from the analytic
sigma_true), the noise draw, and the quadrature subtraction of sigma_noise.  For every field of
data/s5_tau1_1200.npz this recomputes sigma_alpha_clean and the noisy sigma_alpha from the same
seeds (lam, sigma_true and sigma_alpha are asserted equal to the stored values), and prints the
three steps for every run called wrongly reducible at tau_1 = 4 and 8 deg.

Output: data/s5_readout_check.npz.  About 10 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.core import M_ANCH, plan_uniform, merged_run, read_interp, axial  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, draw_lam, calibrate_eps, sigma_true, S5_SEED0 as SEED0  # noqa: E402

LEVELS = (1.0, 4.0)
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "data", "s5_tau1_1200.npz"), allow_pickle=True)
A = d["rows"]
c = {str(k): i for i, k in enumerate(d["cols"])}
stored = {(int(r[c["field"]]), float(r[c["eps_mult"]])): r for r in A}
fields = sorted({int(r[c["field"]]) for r in A})

rec = {k: [] for k in ("field", "lam", "sigma_true", "sa_clean", "sa_x1", "sa_x4")}
prof = []
t0 = time.time()
for j in fields:
    seed = SEED0 + j
    nm, f, sc, a_, b_ = draw_ridge(seed)
    lam = draw_lam(seed)
    r1 = stored[(j, 1.0)]
    assert lam == r1[c["lam"]], (j, lam, r1[c["lam"]])
    eps0 = calibrate_eps(mk(sc, f, a_, b_, 0.0), seed)
    resp = mk(sc, f, a_, b_, lam)
    CX, CY, HX, HY, FC, F0 = build_field(resp, M_ANCH, seed)
    SFC = FC - F0[:, None]
    S = plan_uniform(M_ANCH)
    s_true, _ = sigma_true(resp, CX, CY)
    assert abs(s_true - r1[c["sigma_true"]]) <= 1e-9, (j, s_true, r1[c["sigma_true"]])
    sa_clean = float(axial(read_interp(SFC, S, HX, HY))[1])
    sa_noisy = {}
    for mxx in LEVELS:
        prof_, _pool, _yn, _E = merged_run(SFC, F0, mxx * eps0, np.random.default_rng(seed * 19 + int(mxx * 1e6)))
        sa_noisy[mxx] = float(axial(read_interp(prof_, S, HX, HY))[1])
        ref = stored[(j, mxx)][c["sigma_alpha"]]
        assert abs(sa_noisy[mxx] - ref) <= 1e-9, (j, mxx, sa_noisy[mxx], ref)
    for k, v in (("field", j), ("lam", lam), ("sigma_true", s_true), ("sa_clean", sa_clean), ("sa_x1", sa_noisy[1.0]), ("sa_x4", sa_noisy[4.0])):
        rec[k].append(v)
    prof.append(nm)
R = {k: np.array(v) for k, v in rec.items()}
P = np.array(prof)
print("%d fields in %.0f s; lam, sigma_true and the noisy sigma_alpha of all %d runs match the stored values" % (len(P), time.time() - t0, 2 * len(P)))

idx = {j: i for i, j in enumerate(R["field"])}
for tau in (4.0, 8.0):
    for mxx in LEVELS:
        print("\n== wrongly reducible at tau_1 = %g deg, eps x%g" % (tau, mxx))
        print("   %5s %-9s %6s | %7s %9s %13s %9s %7s | %8s %6s"
              % ("field", "profile", "lam", "s_true", "sa_clean", "sqrt(cl2+sn2)", "sa_noisy", "s_noise", "s_t_hat", "q95"))
        for (j, m), r in sorted(stored.items()):
            if m != mxx:
                continue
            s, q, st, sn, sa = (r[c[k]] for k in ("sigma_true", "q95", "sigma_true_hat", "sigma_noise", "sigma_alpha"))
            if not (s > tau and st <= tau and q < tau):
                continue
            i = idx[j]
            print("   %5d %-9s %6.3f | %7.2f %9.2f %13.2f %9.2f %7.2f | %8.2f %6.2f"
                  % (j, P[i], R["lam"][i], s, R["sa_clean"][i], math.hypot(R["sa_clean"][i], sn), sa, sn, st, q))

print("\nsigma_alpha_clean / sigma_true by profile and sigma_true bin, median (IQR) [n]")
for nm in sorted(set(P)):
    cells = []
    for lo, hi in ((1, 4), (4, 8), (8, 20)):
        b = (P == nm) & (R["sigma_true"] > lo) & (R["sigma_true"] <= hi)
        if b.sum() >= 3:
            r = R["sa_clean"][b] / R["sigma_true"][b]
            cells.append("(%g, %g]: %.2f (%.2f-%.2f) [%d]" % (lo, hi, np.median(r), *np.percentile(r, [25, 75]), b.sum()))
        else:
            cells.append("(%g, %g]: n %d" % (lo, hi, b.sum()))
    print("   %-9s %s" % (nm, " | ".join(cells)))
for tau in (4.0, 8.0):
    near = (R["sigma_true"] > tau) & (R["sigma_true"] <= 1.5 * tau)
    below = near & (R["sa_clean"] < tau)
    print("fields with tau_1 < sigma_true <= 1.5 tau_1 at tau_1 = %g: %d, of which the noise-free read-out is already below tau_1: %d"
          % (tau, near.sum(), below.sum()))

out = os.path.join(HERE, "data", "s5_readout_check.npz")
np.savez(out, profile=P, **R)
print("saved", out)
