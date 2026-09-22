# -*- coding: utf-8 -*-
"""Synthetic response fields for calibration (Methods, "Synthetic response fields").

Two families on Omega = [-2, 2]^2 (log coordinates):
    single ridge + linear term    y = c F(a x1 + b x2) + lam c (-b x1 + a x2)        mk()
    two ridges                    y = c F(a x1 + b x2) + lam c' G(a' x1 + b' x2)     mk2()
lam = 0 is exactly reducible.  F, G are drawn from the five profiles of RIDGE, whose second
entry is the constant c attached to the profile.

The random streams of a field with seed s (s = s0 + j for field j of a design):
    draw_ridge(s)      profile and direction of the main ridge      numpy default_rng(s)
    draw_second(s, ..) profile and direction of the second ridge    default_rng(7919 s + 13)
    build_field(.., s) the M anchors of the run                    torch generator, seed s
    calibration        60 preliminary anchors, seed s + 7777, at the smallest probe
The observation noise and the null simulation are seeded in each script.
"""
import math

import numpy as np
import torch

from .core import FRAC_MIN, SIG_REL, _t

RIDGE = {
    'exp':      (lambda s: np.exp(np.clip(s, -20, 20)), 1.0),
    'softplus': (lambda s: np.log1p(np.exp(-np.abs(s))) + np.maximum(s, 0), 25.0),
    'arcsinh':  (np.arcsinh, 75.0),
    'sinh':     (lambda s: np.sinh(np.clip(s, -20, 20)), 1.0),
    'tanh':     (np.tanh, 200.0),
}
NAMES = list(RIDGE)
BOX = (-2.0, 2.0, -2.0, 2.0)
N_CALIB = 60                    # preliminary anchors of the noise calibration


def build_field(resp, m, seed, cap=None, box=BOX):
    """m anchors with the probe profile at all SIZE candidate angles; as core.build, for a
    response function on the synthetic box.  Same draw order as the benchmarks."""
    x0, x1, y0, y1 = box
    fhx, fhy = (x1 - x0) / 2, (y1 - y0) / 2
    mx, my = FRAC_MIN * fhx, FRAC_MIN * fhy
    g = torch.Generator(); g.manual_seed(seed)
    CX, CY, HX, HY, FC, F0 = [], [], [], [], [], []
    for _ in range(m):
        cx = float(torch.rand(1, generator=g).item()) * (x1 - x0 - 2 * mx) + (x0 + mx)
        cy = float(torch.rand(1, generator=g).item()) * (y1 - y0 - 2 * my) + (y0 + my)
        hx, hy = min(cx - x0, x1 - cx), min(cy - y0, y1 - cy)
        if cap is not None:
            hx, hy = min(hx, cap * fhx), min(hy, cap * fhy)
        FC.append(np.array([resp(cx + hx * math.cos(t), cy + hy * math.sin(t)) for t in _t]))
        F0.append(resp(cx, cy))
        CX.append(cx); CY.append(cy); HX.append(hx); HY.append(hy)
    return tuple(np.array(v) for v in (CX, CY, HX, HY, FC, F0))


def mk(sc, f, a, b, lam):
    """Single ridge with a linear transverse term."""
    def resp(x, y):
        out = sc * float(f(np.array(a * x + b * y)))
        if lam:
            out += lam * sc * float(-b * x + a * y)
        return out
    return resp


def mk2(sc, f, a, b, lam, scg, g, c, d):
    """Single ridge plus a second ridge; lam = 0 is identical to mk(sc, f, a, b, 0)."""
    def resp(x, y):
        out = sc * float(f(np.array(a * x + b * y)))
        if lam:
            out += lam * scg * float(g(np.array(c * x + d * y)))
        return out
    return resp


def draw_ridge(seed):
    """Profile and unit direction of the main ridge: the profile with equal probability, then a
    point uniform in [-1, 1]^2 normalised to unit length; a point of norm below 0.2 is replaced
    by (1, 1) / sqrt 2.  Returns (name, F, c, a, b)."""
    rr = np.random.default_rng(seed)
    nm = NAMES[rr.integers(len(NAMES))]
    f, sc = RIDGE[nm]
    a_, b_ = rr.uniform(-1, 1), rr.uniform(-1, 1)
    n_ = math.hypot(a_, b_)
    if n_ < 0.2:
        a_, b_ = 0.7, 0.7
        n_ = math.hypot(a_, b_)
    return nm, f, sc, a_ / n_, b_ / n_


S5_SEED0, S5_P_EXACT, S5_LAM_LO, S5_LAM_HI = 63300000, 0.25, 0.001, 0.3


def draw_lam(seed, p_exact=S5_P_EXACT, lo=S5_LAM_LO, hi=S5_LAM_HI):
    """lam of a field of the Supplementary Fig. 5 design, from its own stream (7 s + 3): 0 with
    probability p_exact, otherwise log-uniform on [lo, hi]."""
    r2 = np.random.default_rng(seed * 7 + 3)
    return 0.0 if r2.random() < p_exact else float(math.exp(r2.uniform(math.log(lo), math.log(hi))))


def draw_second(seed, a_, b_):
    """Profile and unit direction of the second ridge, from an independent stream: redrawn until
    the norm is at least 0.2 and |a a' + b b'| < 0.9.  Returns (name, G, c', a', b')."""
    r2 = np.random.default_rng(seed * 7919 + 13)
    nm2 = NAMES[r2.integers(len(NAMES))]
    g, scg = RIDGE[nm2]
    while True:
        c_, d_ = r2.uniform(-1, 1), r2.uniform(-1, 1)
        n2 = math.hypot(c_, d_)
        if n2 < 0.2:
            continue
        c_, d_ = c_ / n2, d_ / n2
        if abs(a_ * c_ + b_ * d_) < 0.9:
            break
    return nm2, g, scg, c_, d_


def calibrate_eps(resp0, seed):
    """The noise level of a synthetic field: SIG_REL times the median probe amplitude at the
    smallest probe over N_CALIB preliminary anchors (seed + 7777), on the field with lam = 0.
    Returns nan when the responses are not finite (no such field occurred in the paper)."""
    _q, _w, _e, _r, FCc, F0c = build_field(resp0, N_CALIB, seed + 7777, cap=FRAC_MIN)
    eps0 = SIG_REL * float(np.median(np.abs(FCc - F0c[:, None]).max(1)))
    return eps0 if (np.isfinite(eps0) and eps0 > 0) else float('nan')


def sigma_true(resp, CX, CY, h=1e-4):
    """The analytic reference of stage one: the axial spread (degrees) of the gradient directions
    of the noise-free field at the anchors, by central differences with step h."""
    from .core import axial
    a_tr = np.array([math.atan2((resp(cx, cy + h) - resp(cx, cy - h)) / (2 * h),
                                (resp(cx + h, cy) - resp(cx - h, cy)) / (2 * h)) % math.pi
                     for cx, cy in zip(CX, CY)])
    return float(axial(a_tr)[1]), a_tr
