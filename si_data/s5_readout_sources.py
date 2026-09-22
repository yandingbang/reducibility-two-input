# -*- coding: utf-8 -*-
"""Supplementary Fig. 5 check, part two -- where the read-out bias on tilted fields comes from.

The read-out makes two approximations: the NQ queried angles with linear interpolation of the
zero crossing, and the finite probe (the direction is the normal to the chord from the anchor
to the point of equal response, while sigma_true uses the gradient at the anchor; on a tilted
field the level set bends).  Every field of data/s5_tau1_1200.npz is read noise-free six ways:
NQ angles or the full grid of SIZE angles, with the deployed probes or probes capped at 0.4,
0.1 and 0.02.  The anchors are the same under every cap (asserted).

Output: data/s5_readout_sources.npz.  About 10 min.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.core import M_ANCH, SIZE, plan_uniform, read_interp, axial  # noqa: E402
from framework.synthetic import build_field, mk, draw_ridge, draw_lam, sigma_true, S5_SEED0 as SEED0  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NA = SIZE
#           label,                                probe cap, angles
CONFIGS = (("15 angles, deployed probes",         None,      15),
           ("%d angles, deployed probes" % NA,    None,      NA),
           ("%d angles, probes <= 0.4" % NA,      0.20,      NA),
           ("%d angles, probes <= 0.1" % NA,      0.05,      NA),
           ("%d angles, probes <= 0.02" % NA,     0.01,      NA),
           ("15 angles, probes <= 0.02",          0.01,      15))
SHORT = ("15/deployed", "%d/deployed" % NA, "%d/0.4" % NA, "%d/0.1" % NA, "%d/0.02" % NA, "15/0.02")

d = np.load(os.path.join(HERE, "data", "s5_tau1_1200.npz"), allow_pickle=True)
A = d["rows"]
c = {str(k): i for i, k in enumerate(d["cols"])}
stored = {(int(r[c["field"]]), float(r[c["eps_mult"]])): r for r in A}
fields = sorted({int(r[c["field"]]) for r in A})
prev_path = os.path.join(HERE, "data", "s5_readout_check.npz")
prev_clean = None
if os.path.exists(prev_path):
    prev = np.load(prev_path, allow_pickle=True)
    prev_clean = dict(zip(prev["field"].astype(int), prev["sa_clean"]))

S15 = plan_uniform(M_ANCH)
S90 = [list(range(SIZE)) for _ in range(M_ANCH)]
CAPS = [None] + sorted({cfg[1] for cfg in CONFIGS if cfg[1] is not None})

print("read-out sources: %d fields, read-outs: %s" % (len(fields), ", ".join(SHORT)), flush=True)
SA = np.empty((len(fields), len(CONFIGS)))
ST_, LAM_, PROF = np.empty(len(fields)), np.empty(len(fields)), []
t0 = time.time()
for i, j in enumerate(fields):
    seed = SEED0 + j
    nm, f, sc, a_, b_ = draw_ridge(seed)
    lam = draw_lam(seed)
    assert lam == stored[(j, 1.0)][c["lam"]], j
    resp = mk(sc, f, a_, b_, lam)
    built = {}
    for cap in CAPS:
        CX, CY, HX, HY, FC, F0 = build_field(resp, M_ANCH, seed, cap=cap)
        built[cap] = (CX, CY, HX, HY, FC - F0[:, None])
    CX, CY = built[None][0], built[None][1]
    for cap in CAPS:
        assert np.array_equal(built[cap][0], CX) and np.array_equal(built[cap][1], CY), (j, cap)
    s_true, _ = sigma_true(resp, CX, CY)
    assert abs(s_true - stored[(j, 1.0)][c["sigma_true"]]) <= 1e-9, j
    for k, (_, cap, nang) in enumerate(CONFIGS):
        _, _, HX, HY, SFC = built[cap]
        SA[i, k] = float(axial(read_interp(SFC, S15 if nang == 15 else S90, HX, HY))[1])
    if prev_clean is not None:
        assert abs(SA[i, 0] - prev_clean[j]) <= 1e-9, (j, SA[i, 0], prev_clean[j])
    ST_[i], LAM_[i] = s_true, lam
    PROF.append(nm)
    if (i + 1) % 200 == 0:
        print("  field %4d  %.0f s" % (i + 1, time.time() - t0), flush=True)
P = np.array(PROF)
print("%d fields in %.0f s; anchors identical under every cap; sigma_true reproduces the stored values" % (len(P), time.time() - t0))

z = LAM_ == 0
print("\nexactly reducible fields (%d): noise-free sigma_alpha in deg, median / 95th percentile / max" % z.sum())
for k, name in enumerate(SHORT):
    v = SA[z, k]
    print("    %-12s %.3f / %.3f / %.3f" % (name, np.median(v), np.percentile(v, 95), v.max()))

BINS = ((1, 4), (4, 8), (8, 20))
print("\ntilted fields: sigma_alpha_clean / sigma_true, median (IQR), and share with |ratio - 1| > 0.1")
print("    %-12s %s" % ("", " | ".join("sigma_true (%g, %g], n %d" % (lo, hi, np.sum((ST_ > lo) & (ST_ <= hi))) for lo, hi in BINS)))
for k, name in enumerate(SHORT):
    cells = []
    for lo, hi in BINS:
        b = (ST_ > lo) & (ST_ <= hi)
        r = SA[b, k] / ST_[b]
        cells.append("%.2f (%.2f-%.2f) %3.0f%%" % (np.median(r), *np.percentile(r, [25, 75]), 100 * np.mean(np.abs(r - 1) > 0.1)))
    print("    %-12s %s" % (name, " | ".join("%-25s" % s for s in cells)))

print("\nby profile, sigma_true > 4 deg: median sigma_alpha_clean / sigma_true")
print("    %-9s %5s " % ("profile", "n") + " ".join("%12s" % s for s in SHORT))
for nm in sorted(set(P)):
    b = (P == nm) & (ST_ > 4)
    if b.sum() >= 3:
        print("    %-9s %5d " % (nm, b.sum()) + " ".join("%12.2f" % np.median(SA[b, k] / ST_[b]) for k in range(len(SHORT))))

print("\nnear the boundary, noise-free:")
for tau in (4.0, 8.0):
    above = (ST_ > tau) & (ST_ <= 1.5 * tau)
    below = (ST_ < tau) & (ST_ >= tau / 1.5)
    print("  tau_1 %g: of %d fields with tau_1 < sigma_true <= 1.5 tau_1, read below tau_1: %s"
          % (tau, above.sum(), "  ".join("%s %d" % (SHORT[k], np.sum(above & (SA[:, k] < tau))) for k in range(len(SHORT)))))
    print("  tau_1 %g: of %d fields with tau_1/1.5 <= sigma_true < tau_1, read above tau_1: %s"
          % (tau, below.sum(), "  ".join("%s %d" % (SHORT[k], np.sum(below & (SA[:, k] > tau))) for k in range(len(SHORT)))))

idx = {j: i for i, j in enumerate(fields)}
for tau in (4.0, 8.0):
    wrong = sorted({j for (j, m), r in stored.items() if r[c["sigma_true"]] > tau and r[c["sigma_true_hat"]] <= tau and r[c["q95"]] < tau})
    print("\nwrongly reducible at tau_1 = %g deg (either noise level), noise-free read-outs:" % tau)
    print("    %5s %-9s %6s  " % ("field", "profile", "s_true") + " ".join("%12s" % s for s in SHORT))
    for j in wrong:
        i = idx[j]
        print("    %5d %-9s %6.2f  " % (j, P[i], ST_[i]) + " ".join("%12.2f" % SA[i, k] for k in range(len(SHORT))))

out = os.path.join(HERE, "data", "s5_readout_sources.npz")
np.savez(out, configs=np.array([cfg[0] for cfg in CONFIGS]), short=np.array(SHORT), field=np.array(fields), lam=LAM_, profile=P, sigma_true=ST_, sa_clean=SA)
print("\nsaved", out)
