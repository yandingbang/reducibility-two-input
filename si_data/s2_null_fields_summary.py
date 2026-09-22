# -*- coding: utf-8 -*-
"""Summary of data/s2_null_fields_<N>.npz (Supplementary Fig. 2b and its caption).

For each noise level: the share of runs above their own q95 and the shares with p <= 1, 5,
10 and 20%, then the 5% rate split by the ratio q95 / read-out floor and by ridge profile,
and the deployed stage-one verdicts at tau_1 = 4 deg.  Clopper-Pearson 95% intervals.
Prints only.  Usage: python s2_null_fields_summary.py [N_FIELD, default 800]
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.core import v1, TAU1                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
d = np.load(os.path.join(HERE, "data", "s2_null_fields_%d.npz" % N), allow_pickle=True)
A, prof = d["rows"], np.array([str(s) for s in d["profile"]])
c = {str(name): i for i, name in enumerate(d["cols"])}
print("settings: N_MC %d, N_FIELDSIM %d, N_MERGE %d, M %d; %d runs"
      % (int(d["n_mc"]), int(d["n_fieldsim"]), int(d["n_merge"]), int(d["m_anch"]), len(A)))


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi


def rate(label, B):
    n, k = len(B), int(B[:, c["exceed"]].sum())
    lo, hi = cp(k, n)
    return "   %-26s n %4d  above q95 %3d = %5.2f%%  (95%% CI %5.2f-%5.2f%%)" % (
        label, n, k, 100.0 * k / max(n, 1), lo, hi)


for m in sorted(set(A[:, c["eps_mult"]])):
    sel = A[:, c["eps_mult"]] == m
    B, P = A[sel], prof[sel]
    n = len(B)
    print("\n== noise eps x%g" % m)
    print(rate("all fields", B))
    for a in (0.01, 0.05, 0.10, 0.20):
        k = int(np.sum(B[:, c["p"]] <= a))
        lo, hi = cp(k, n)
        print("   p <= %2d%%: %6.2f%%  (95%% CI %5.2f-%5.2f%%)" % (round(100 * a), 100.0 * k / n, lo, hi))
    print("   sigma_true_hat clipped to 0 in %.1f%% of runs; median q95 %.3f, median floor %.3f"
          % (100 * np.mean(B[:, c["st"]] == 0), np.median(B[:, c["q95"]]), np.median(B[:, c["floor"]])))
    ratio = B[:, c["q95"]] / np.maximum(B[:, c["floor"]], 1e-12)
    print("   by q95 / read-out floor:")
    for lo_r, hi_r in ((0, 3), (3, 10), (10, np.inf)):
        k = (ratio >= lo_r) & (ratio < hi_r)
        if k.any():
            print(rate("[%g, %g)" % (lo_r, hi_r), B[k]))
    print("   by ridge profile:")
    for nm in sorted(set(P)):
        print(rate(nm, B[P == nm]))
    verdicts = [v1(st, TAU1, q) for st, q in zip(B[:, c["st"]], B[:, c["q95"]])]
    print("   stage-one verdicts at tau_1 = %g deg: %s" % (
        TAU1, ", ".join("%s %d" % (v, verdicts.count(v)) for v in ("REDUCIBLE", "NOT-RED", "UNRESOLVED"))))
