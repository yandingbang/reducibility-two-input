# -*- coding: utf-8 -*-
"""Supplementary Fig. 5 and Table 3 numbers -- the stage-one verdict against the truth, swept over tau_1.

Reads data/s5_tau1_<N_FIELD>.npz.  At each tau_1 every run is scored with the deployed rule v1;
the truth at tau_1 is the analytic sigma_true: reducible when sigma_true < tau_1.  Each error is
normalised by its own population:
    false NOT-RED = NOT-RED   among runs with sigma_true < tau_1
    missed        = REDUCIBLE among runs with sigma_true > tau_1
    UNRESOLVED    = UNRESOLVED among all runs
One run per field, so the runs are independent: Clopper-Pearson 95% intervals.  Also printed:
where the errors sit relative to tau_1, sigma_true_hat / sigma_true by bin and by profile.

Env: N_FIELD (1200).  Prints only.
"""
import os
import sys
from collections import Counter

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.core import v1                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_FIELD = int(os.environ.get("N_FIELD", 1200))
TAUS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
LEVELS = (1.0, 4.0)
NEAR = 1.5


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi


def rate(k, n):
    if n == 0:
        return "%22s" % "-"
    lo, hi = cp(k, n)
    return "%4d/%-4d %5.2f%% [%.1f-%.1f]" % (k, n, 100.0 * k / n, lo, hi)


def verdicts(st, q95, tau):
    """v1 applied to every run: (NOT-RED, REDUCIBLE, UNRESOLVED) masks"""
    v = np.array([v1(s, tau, q) for s, q in zip(st, q95)])
    return v == "NOT-RED", v == "REDUCIBLE", v == "UNRESOLVED"


d = np.load(os.path.join(HERE, "data", "s5_tau1_%d.npz" % N_FIELD), allow_pickle=True)
A = d["rows"]
P = d["profile"]
c = {str(k): i for i, k in enumerate(d["cols"])}
print("s5_tau1_%d.npz: %d runs, N_MERGE %d, N_MC %d, N_FIELDSIM %d"
      % (N_FIELD, len(A), int(d["n_merge"]), int(d["n_mc"]), int(d["n_fieldsim"])))


def level(mxx):
    k = A[:, c["eps_mult"]] == mxx
    B = A[k]
    return (k, B[:, c["sigma_true"]], B[:, c["q95"]], B[:, c["sigma_true_hat"]], B[:, c["lam"]], B[:, c["field"]].astype(int))


for mxx in LEVELS:
    k, s, q, st, lam, fld = level(mxx)
    print("\n== eps x%g: %d runs, exactly reducible %d; q95 median %.2f deg (p10 %.2f, p90 %.2f)"
          % (mxx, len(s), int(np.sum(lam == 0)), np.median(q), *np.percentile(q, [10, 90])))
    print("%6s  %-30s %-30s %-30s %-18s %-18s"
          % ("tau_1", "UNRESOLVED (all)", "false NOT-RED (s<tau)", "missed (s>tau)", "s/tau false n-r", "s/tau missed"))
    for t in TAUS:
        nr, rd, un = verdicts(st, q, t)
        below, above = s < t, s > t
        fn, ms = nr & below, rd & above
        wf = "med %.2f min %.2f" % (np.median(s[fn] / t), np.min(s[fn] / t)) if fn.any() else "-"
        wm = "med %.2f max %.2f" % (np.median(s[ms] / t), np.max(s[ms] / t)) if ms.any() else "-"
        print("%6.2f  %-30s %-30s %-30s %-18s %-18s"
              % (t, rate(int(un.sum()), len(s)), rate(int(fn.sum()), int(below.sum())), rate(int(ms.sum()), int(above.sum())), wf, wm))

print("\n\nWHERE THE ERRORS SIT ('near' = sigma_true within a factor %g of tau_1)" % NEAR)
for mxx in LEVELS:
    k, s, q, st, lam, fld = level(mxx)
    z = lam == 0
    print("\n== eps x%g: exactly reducible fields with sigma_true_hat above their own q95: %s"
          % (mxx, rate(int(np.sum(st[z] > q[z])), int(z.sum()))))
    print("%6s  %-30s %-30s %-30s %-30s %s"
          % ("tau_1", "missed, near", "missed, s > 1.5 tau", "false NOT-RED, near", "false NOT-RED, s < tau/1.5", "near share of s > tau"))
    for t in TAUS:
        nr, rd, un = verdicts(st, q, t)
        an, af = (s > t) & (s <= NEAR * t), s > NEAR * t
        bn, bf = (s < t) & (s >= t / NEAR), s < t / NEAR
        print("%6.2f  %-30s %-30s %-30s %-30s %.0f%%"
              % (t, rate(int((rd & an).sum()), int(an.sum())), rate(int((rd & af).sum()), int(af.sum())),
                 rate(int((nr & bn).sum()), int(bn.sum())), rate(int((nr & bf).sum()), int(bf.sum())),
                 100.0 * an.sum() / max(int((s > t).sum()), 1)))
    print("   sigma_true_hat / sigma_true by sigma_true bin, median (IQR):")
    for lo, hi in ((0.25, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 20)):
        b = (s > lo) & (s <= hi)
        r = st[b] / s[b]
        print("     (%5.2f, %5.2f]  n %4d   %.2f (%.2f-%.2f)" % (lo, hi, b.sum(), np.median(r), *np.percentile(r, [25, 75])))
    print("   sigma_true_hat / sigma_true where sigma_true > 4 deg, by ridge profile, median (IQR):")
    pr = P[k]
    for nm in sorted(set(pr)):
        b = (pr == nm) & (s > 4)
        if b.any():
            r = st[b] / s[b]
            print("     %-9s n %3d   %.2f (%.2f-%.2f)" % (nm, b.sum(), np.median(r), *np.percentile(r, [25, 75])))

print("\nMissed fields shared by the two noise levels")
for t in (4.0, 8.0):
    got = {}
    for mxx in LEVELS:
        k, s, q, st, lam, fld = level(mxx)
        m = verdicts(st, q, t)[1] & (s > t)
        got[mxx] = dict(zip(fld[m], P[k][m]))
    both = set(got[1.0]) & set(got[4.0])
    print("  tau_1 %g deg: x1 %d fields, x4 %d fields, in both %d; profiles at x1 %s"
          % (t, len(got[1.0]), len(got[4.0]), len(both), dict(Counter(got[1.0].values()))))
