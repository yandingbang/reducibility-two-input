# -*- coding: utf-8 -*-
"""Supplementary Note 1 -- direction statistics do not determine collapse accuracy.

Three fields on the unit square,
    y_n(w1, w2) = w2 + (A / 2 pi n) sin(2 pi n w1),   A = 1.481,   n = 1, 4, 16,
sampled on the lattice w1 = i / 49 (i = 0..48), w2 = (j + 1/2) / 8 (j = 0..7).  Because
gcd(n, 49) = 1, the three fields carry the same multiset of gradient vectors, so every
permutation-invariant statistic of the gradients (sigma_alpha, the active-subspace matrix)
is identical, while the one-dimensional collapse residual differs by an order of magnitude.

Printed: the pointwise sigma_alpha and alpha_bar (identical to machine precision), and the
collapse residual rho of framework.core.rho_poly (degree ladder included) minimised over
    360 directions on [0, pi)                 the best collapse each field admits
    the stage-two window of the framework     alpha_bar +/- 3 sigma_alpha / sqrt(M), 21 candidates
    the coordinate w2 alone
Output: data/sigma_alpha_counterexample.npz.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from framework.core import axial, rho_poly, K_WIN, N_WIN                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K_W1, N_W2 = 49, 8              # 49 is coprime to every n below
NN = (1, 4, 16)
AMP = 1.481
N_SCAN = 360

W1 = np.arange(K_W1) / K_W1
W2 = (np.arange(N_W2) + 0.5) / N_W2
CX, CY = np.meshgrid(W1, W2, indexing='ij')
CX, CY = CX.ravel(), CY.ravel()
M = CX.size


def field(n):
    return CY + (AMP / (2 * math.pi * n)) * np.sin(2 * math.pi * n * CX)


def pointwise_dirs(n):
    """direction of the gradient (A cos(2 pi n w1), 1) at each lattice point, mod pi"""
    return np.arctan2(np.ones(M), AMP * np.cos(2 * math.pi * n * CX)) % math.pi


def rho_at(y, t):
    return rho_poly(math.cos(t) * CX + math.sin(t) * CY, y)


def rho_min(y, angles):
    return min((rho_at(y, t), t) for t in angles)


print('M = %d lattice points (%d x %d), A = %g, n = %s' % (M, K_W1, N_W2, AMP, NN))
sets = [np.sort(pointwise_dirs(n)) for n in NN]
print('sorted direction multisets, max difference to n = 1: %s rad'
      % ', '.join('%.1e' % np.abs(sets[0] - s).max() for s in sets[1:]))

rows = []
print('\n%4s %14s %12s | %10s %8s | %10s | %10s' % ('n', 'sigma_alpha', 'alpha_bar', 'rho global', 'at deg',
                                                    'rho window', 'rho on w2'))
for n in NN:
    y = field(n)
    ab, sa = axial(pointwise_dirs(n))
    r_glob, t_glob = rho_min(y, np.linspace(0, math.pi, N_SCAN, endpoint=False))
    half = K_WIN * math.radians(sa) / math.sqrt(M)
    r_win, t_win = rho_min(y, np.linspace(ab - half, ab + half, N_WIN))
    r_w2 = rho_at(y, math.pi / 2)
    rows.append((n, sa, math.degrees(ab), r_glob, math.degrees(t_glob) % 180, r_win, math.degrees(t_win) % 180, r_w2))
    print('%4d %14.6f %12.4f | %10.4f %8.1f | %10.4f | %10.4f' % (n, sa, math.degrees(ab), r_glob,
                                                                   math.degrees(t_glob) % 180, r_win, r_w2))
R = np.array(rows)
print('\nsigma_alpha identical to %.1e deg; rho spans %.1fx (%.3f -> %.3f)'
      % (np.ptp(R[:, 1]), R[0, 3] / R[-1, 3], R[0, 3], R[-1, 3]))

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
out = os.path.join(HERE, 'data', 'sigma_alpha_counterexample.npz')
np.savez(out, n=np.array(NN), amp=AMP, CX=CX, CY=CY, sigma_pointwise=R[:, 1], alpha_bar_deg=R[:, 2],
         rho=R[:, 3], rho_dir_deg=R[:, 4], rho_window=R[:, 5], rho_window_dir_deg=R[:, 6], rho_w2=R[:, 7],
         s_y=np.array([float(np.std(field(n), ddof=1)) for n in NN]))
print('saved', out)
