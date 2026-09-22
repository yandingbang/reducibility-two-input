# -*- coding: utf-8 -*-
"""Supplementary Fig. 6 data -- the read-out error on the eight benchmarks, term by term.

The anchors of each benchmark stay where the protocol puts them (seed RSEED + unit, 40 anchors);
only the probe semi-axes are multiplied by lam = 1, 1/2, 1/4, 1/8, 1/16, 1/40.  Noise-free.
Each profile is read at the NQ queried angles and on the full grid of SIZE angles; the truth
is the gradient direction at each anchor by central differences (step 1e-5, checked at 1e-4).
Per anchor, NQ-angle read-out - truth = (NQ-angle - full grid) + (full grid - truth): the
first part is the interpolation term, the second the chord term.

Output: data/s6_probe_scale.npz.  About 5 min (the wound-closure solves dominate).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.benchmarks import CASES, UNITS                     # noqa: E402
from framework.core import M_ANCH, FRAC_MIN, RSEED, SIZE, _t, build, plan_uniform, read_interp, axial  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAMS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.025)
H_TRUTH = (1e-5, 1e-4)


def build_scaled(key, sub, m, seed, lam):
    """framework.core.build with the probe semi-axes multiplied by lam; same anchors"""
    import torch
    cfg = CASES[key]
    x0, x1, y0, y1 = cfg["boxes"][0] if sub is None else cfg["boxes"][sub]
    resp = cfg["resp"]
    fhx, fhy = (x1 - x0) / 2, (y1 - y0) / 2
    mx, my = FRAC_MIN * fhx, FRAC_MIN * fhy
    g = torch.Generator()
    g.manual_seed(seed)
    CX, CY, HX, HY, FC, F0 = [], [], [], [], [], []
    for _ in range(m):
        cx = float(torch.rand(1, generator=g).item()) * (x1 - x0 - 2 * mx) + (x0 + mx)
        cy = float(torch.rand(1, generator=g).item()) * (y1 - y0 - 2 * my) + (y0 + my)
        hx, hy = min(cx - x0, x1 - cx) * lam, min(cy - y0, y1 - cy) * lam
        FC.append(np.array([resp(cx + hx * math.cos(t), cy + hy * math.sin(t)) for t in _t]))
        F0.append(resp(cx, cy))
        CX.append(cx); CY.append(cy); HX.append(hx); HY.append(hy)
    return tuple(np.array(v) for v in (CX, CY, HX, HY, FC, F0))


def dwrap(a):
    """axial angle difference in radians -> degrees in [-90, 90)"""
    return (np.degrees(a) + 90.0) % 180.0 - 90.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


S15 = plan_uniform(M_ANCH)
S90 = [list(range(SIZE)) for _ in range(M_ANCH)]
ALL = "%d angles" % SIZE
nU, nL = len(UNITS), len(LAMS)
TRUTH = np.zeros((nU, len(H_TRUTH)))
ATR = np.zeros((nU, M_ANCH))
ANG15, ANG90 = np.zeros((nU, nL, M_ANCH)), np.zeros((nU, nL, M_ANCH))
SIG15, SIG90 = np.zeros((nU, nL)), np.zeros((nU, nL))
CXa, CYa, HXa, HYa = (np.zeros((nU, M_ANCH)) for _ in range(4))
DTR = np.zeros(nU)
names, reds = [], []
print("probe scale on the benchmarks: lams %s, 15 angles and %s, noise-free" % (LAMS, ALL), flush=True)
t0 = time.time()
for ui, (name, key, sub, red) in enumerate(UNITS):
    seed = RSEED + ui
    resp = CASES[key]["resp"]
    names.append(name)
    reds.append(bool(red))
    for li, lam in enumerate(LAMS):
        CX, CY, HX, HY, FC, F0 = build_scaled(key, sub, M_ANCH, seed, lam)
        if li == 0:
            for a, b in zip(build(key, sub, M_ANCH, seed), (CX, CY, HX, HY, FC, F0)):
                assert np.array_equal(a, b), name          # lam = 1 is the deployed build
            CXa[ui], CYa[ui], HXa[ui], HYa[ui] = CX, CY, HX, HY
            trs = []
            for h in H_TRUTH:
                trs.append(np.array([math.atan2((resp(cx, cy + h) - resp(cx, cy - h)) / (2 * h),
                                                (resp(cx + h, cy) - resp(cx - h, cy)) / (2 * h)) % math.pi
                                     for cx, cy in zip(CX, CY)]))
            ATR[ui] = trs[0]
            TRUTH[ui] = [axial(a)[1] for a in trs]
            DTR[ui] = float(np.max(np.abs(dwrap(trs[1] - trs[0]))))
        else:
            assert np.array_equal(CX, CXa[ui]) and np.array_equal(CY, CYa[ui]), (name, lam)
        SFC = FC - F0[:, None]
        ANG15[ui, li] = read_interp(SFC, S15, HX, HY)
        ANG90[ui, li] = read_interp(SFC, S90, HX, HY)
        SIG15[ui, li], SIG90[ui, li] = axial(ANG15[ui, li])[1], axial(ANG90[ui, li])[1]
    print("  %-22s done, %.0f s" % (name, time.time() - t0), flush=True)

print("\nwiring: lam = 1 equals the deployed build() on all 8 benchmarks; anchors identical at every lam")
print("truth, step 1e-5 against 1e-4: largest per-anchor direction difference %.2e deg" % DTR.max())

print("\nsigma read-out, deg (truth; then 15 angles / %s at lam = %s)" % (ALL, ", ".join("%g" % l for l in LAMS)))
for u, n in enumerate(names):
    print("  %-22s truth %8.4f" % (n, TRUTH[u, 0]))
    print("  %-22s   15 angles  %s" % ("", "  ".join("%8.4f" % v for v in SIG15[u])))
    print("  %-22s   %-10s %s" % ("", ALL, "  ".join("%8.4f" % v for v in SIG90[u])))

print("\nper-anchor RMS error, deg: chord = full-grid read-out - truth; interpolation = 15-angle - full-grid")
for u, n in enumerate(names):
    ch = [rms(dwrap(ANG90[u, l] - ATR[u])) for l in range(nL)]
    it = [rms(dwrap(ANG15[u, l] - ANG90[u, l])) for l in range(nL)]
    print("  %-22s chord         %s" % (n, "  ".join("%8.4f" % v for v in ch)))
    print("  %-22s   ratio lam/2  %s" % ("", "  ".join("%8.2f" % (ch[l] / max(ch[l + 1], 1e-12)) for l in range(nL - 1))))
    print("  %-22s interpolation %s" % ("", "  ".join("%8.4f" % v for v in it)))
    print("  %-22s   ratio lam/2  %s" % ("", "  ".join("%8.2f" % (it[l] / max(it[l + 1], 1e-12)) for l in range(nL - 1))))

print("\nrelative departure of the sigma read-out from the truth, where the truth exceeds 1 deg")
for u, n in enumerate(names):
    if TRUTH[u, 0] > 1:
        print("  %-22s 15 angles %s | %s %s" % (
            n, " ".join("%+6.1f%%" % (100 * (SIG15[u, l] / TRUTH[u, 0] - 1)) for l in range(nL)),
            ALL, " ".join("%+6.1f%%" % (100 * (SIG90[u, l] / TRUTH[u, 0] - 1)) for l in range(nL))))

out = os.path.join(HERE, "data", "s6_probe_scale.npz")
np.savez(out, names=np.array(names), red=np.array(reds), lams=np.array(LAMS), h_truth=np.array(H_TRUTH),
         truth=TRUTH, a_true=ATR, ang15=ANG15, ang90=ANG90, sig15=SIG15, sig90=SIG90,
         cx=CXa, cy=CYa, hx=HXa, hy=HYa, rseed=RSEED, m_anch=M_ANCH, n_all=SIZE)
print("\nsaved", out, "in %.0f s" % (time.time() - t0))
