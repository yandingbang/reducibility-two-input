# -*- coding: utf-8 -*-
"""The framework as deployed: protocol constants, read-out, stage one, stage two, verdicts.

The functions are the in-service ones of deploy_stop/framework_pipeline.py, copied verbatim;
only the branches that belonged to earlier protocol generations were removed (nothing here
depends on them).  The order of random draws is unchanged, so every result of the paper is
reproduced bit for bit from the seeds recorded in the scripts.

Protocol (Methods, "Evaluation protocol"):
    M_ANCH   anchors per run, drawn uniformly in the regime shrunk by the margin FRAC_MIN
    probe    half-ellipse around each anchor, semi-axes as large as fit in the regime
    NQ       queried angles per anchor, t_j = j pi / (NQ - 1), the same at every anchor
    N_MERGE  repeats of every evaluation
    N_MC     perturbed profiles per anchor in the null simulation
    N_FIELDSIM  synthetic runs for the null distribution of sigma_true_hat
    TAU1, TAU   the two tolerances (degrees, and a fraction of s_y)

SIZE is the fine grid of candidate angles; the NQ queried angles are its indices
0, 7, ..., 98.  The fine grid itself is used only by Supplementary Fig. 6.
"""
import math
import os

import numpy as np
import torch

# ------------------------------------------------------------------ protocol
M_ANCH = 40
NQ = 15
FRAC_MIN = 0.20                 # anchor margin; also the smallest probe
SIG_REL = 0.05                  # benchmark noise: eps = 5% of the median amplitude at h_min
TAU = 0.05                      # stage-two tolerance
DEGREES = [2, 3, 5, 8, 12, 16, 20]
K_WIN, N_WIN = 3.0, 21          # stage-two window: alpha_bar +/- K_WIN sigma_alpha / sqrt(M), N_WIN candidates
RSEED = 20260918                # benchmark seeds: RSEED + unit index
SIZE = 99                       # fine grid of candidate angles on [0, pi]; step pi / 98
_t = np.linspace(0, np.pi, SIZE)

TAU1 = float(os.environ.get('TAU1', 4.0))
N_MERGE = int(os.environ.get('N_MERGE', 6))
N_MC = int(os.environ.get('N_MC', 1200))
N_FIELDSIM = int(os.environ.get('N_FIELDSIM', 6400))


# ------------------------------------------------------------------ anchors and probes
def build(key, sub, m, seed, cap=None):
    """m anchors of a benchmark regime, with the probe profile at all SIZE candidate angles.

    Returns CX, CY (anchor coordinates), HX, HY (probe semi-axes), FC (m x SIZE responses on
    the probe) and F0 (m responses at the anchors).  `cap` limits the semi-axes to cap times
    the half-width; the noise calibration uses cap = FRAC_MIN (the smallest probe).
    """
    from .benchmarks import CASES
    cfg = CASES[key]
    x0, x1, y0, y1 = cfg['boxes'][0] if sub is None else cfg['boxes'][sub]
    resp = cfg['resp']
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


def calibrate_eps(key, sub, seed, n_prelim=200):
    """The benchmark noise level: SIG_REL times the median probe amplitude at the smallest probe,
    over n_prelim preliminary anchors drawn with their own seed (seed + 7777 in every script)."""
    _cx, _cy, _hx, _hy, FCc, F0c = build(key, sub, n_prelim, seed + 7777, cap=FRAC_MIN)
    return SIG_REL * float(np.median(np.abs(FCc - F0c[:, None]).max(1)))


def plan_uniform(m):
    """The query plan: NQ equispaced angles on the fine grid, indices 0, 7, ..., SIZE - 1,
    identical at every anchor.  Depends on no data."""
    u = sorted(set(int(round(v)) for v in np.linspace(0, SIZE - 1, NQ)))
    return [list(u) for _ in range(m)]


def merged_run(SFC, F0, eps, rng, S, n=None):
    """One run of the protocol: n repeats of every evaluation with independent errors.

    The evaluated points are the NQ queried angles of the plan S (the same at every anchor)
    and the anchor itself: M (NQ + 1) points, each evaluated n times.  E has shape
    (n, M, NQ + 1), its last column the anchor; nothing is drawn for unqueried angles.  Returns
        prof  the averaged signed profile  s_ik = mean_r y(p_ik) - mean_r y(c_i), M x SIZE with
              the unqueried columns set to nan (only the queried ones exist)
        pool  the residual pool: the M (NQ + 1) n within-repeat residuals, times sqrt(n / (n - 1))
        yn    the averaged anchor responses                                       (M,)
        E     the errors, (n, M, NQ + 1)
    The Gaussian errors are the generating model of the benchmarks and synthetic fields; the
    framework itself uses only prof, pool and yn.
    """
    n = N_MERGE if n is None else n
    idx = np.array(sorted(S[0]))
    assert all(sorted(Si) == list(idx) for Si in S), "the plan must be the same at every anchor"
    E = rng.normal(0.0, eps, (n, SFC.shape[0], idx.size + 1))
    prof = np.full(SFC.shape, np.nan)
    prof[:, idx] = SFC[:, idx] + E[:, :, :-1].mean(0) - E[:, :, -1:].mean(0)
    pool = ((E - E.mean(0, keepdims=True)) * math.sqrt(n / (n - 1.0))).ravel()
    yn = F0 + E[:, :, -1].mean(0)
    return prof, pool, yn, E


# ------------------------------------------------------------------ the read-out
def readout(t, hx, hy):
    """Direction of the level set from the zero crossing t of the probe profile: the normal of
    the probe ellipse at t, folded to [0, pi)."""
    return math.atan2(-hx * math.cos(t), hy * math.sin(t)) % math.pi


def wrap(x):
    return (x + math.pi / 2) % math.pi - math.pi / 2


def interp_cross(sig, tk):
    """Interpolated zero crossing of the signed profile; falls back to argmin |s| when the
    profile has no sign change."""
    o = np.argsort(tk)
    t, s = tk[o], sig[o]
    sgn = np.sign(s)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return float(t[int(np.argmin(np.abs(s)))])
    k = idx[int(np.argmin(np.abs(s[idx]) + np.abs(s[idx + 1])))]
    a, b = abs(s[k]), abs(s[k + 1])
    return float(t[k] + a / max(a + b, 1e-300) * (t[k + 1] - t[k]))


def read_interp(SFC, S, HX, HY):
    """The direction read at every anchor from its profile at the queried angles S[i]."""
    return np.array([readout(interp_cross(SFC[i, np.array(sorted(Si))],
                                          _t[np.array(sorted(Si))]), HX[i], HY[i])
                     for i, Si in enumerate(S)])


def axial(ang):
    """Axial mean direction and axial spread (degrees) of directions modulo pi."""
    z = np.mean(np.exp(2j * np.asarray(ang)))
    return ((0.5 * np.angle(z)) % math.pi,
            math.degrees(0.5 * math.sqrt(max(-2.0 * math.log(max(abs(z), 1e-16)), 0.0))))


# ------------------------------------------------------------------ stage one
def null_sim(pilot, S, HX, HY, eps, rng, n_anchor, n_field, pool=None, nrep=1,
             return_null=False):
    """sigma_noise and q95 by simulation on the null hypothesis sigma_true = 0.

    For each anchor, n_anchor perturbed copies of its averaged profile are read; the
    perturbation is the difference of two means of nrep residuals drawn with replacement
    from the pool, one mean per queried angle and one shared by the whole anchor, so that
    the simulated profile has the covariance of the observed one.  phi_i is the mean of
    exp(2i delta) over the copies; sigma_noise^2 = -1/2 ln |mean_i phi_i|.  q95 is the 95th
    percentile of sigma_true_hat over n_field synthetic runs assembled from the per-anchor
    deviations.  `eps` is not used when a pool is given (kept for the call signature).

    With return_null, also returns the per-anchor phi_i and the n_field null draws of
    sigma_alpha (Supplementary Figs. 1b, 2 and Table 4 need them); the draws are the same.
    """
    assert pool is not None, "the deployed framework resamples the residual pool"
    m = len(S)
    dev = np.empty((m, n_field))
    phi = []
    for i, Si in enumerate(S):
        Sa = np.array(sorted(Si)); tk = _t[Sa]; s = pilot[i, Sa]
        a0 = readout(interp_cross(s, tk), HX[i], HY[i])
        e = (rng.choice(pool, (n_anchor, tk.size, nrep)).mean(-1)
             - rng.choice(pool, (n_anchor, 1, nrep)).mean(-1))
        d = np.array([wrap(readout(interp_cross(s + e[r], tk), HX[i], HY[i]) - a0)
                      for r in range(n_anchor)])
        phi.append(np.mean(np.exp(2j * d)))
        dev[i] = rng.choice(d, size=n_field, replace=True)
    zbar = float(np.abs(np.mean(phi)))
    sn = math.degrees(math.sqrt(max(-0.5 * math.log(max(zbar, 1e-16)), 0.0)))
    sa_null = np.array([axial(dev[:, r])[1] for r in range(n_field)])
    q95 = float(np.percentile(np.sqrt(np.maximum(sa_null ** 2 - sn ** 2, 0.0)), 95))
    if return_null:
        return sn, q95, np.array(phi), sa_null
    return sn, q95


def sigma_true_hat(sa, sn):
    """The structural spread: the noise floor taken off in quadrature, truncated at zero."""
    return math.sqrt(max(sa ** 2 - sn ** 2, 0.0))


# ------------------------------------------------------------------ stage two
def rho_poly(u, y):
    """Collapse residual of y against the group coordinate u: the residual standard deviation
    of a polynomial fit, normalised by s_y, on the degree ladder DEGREES (the next degree is
    taken only if it lowers the residual by more than 5%)."""
    def at(deg):
        uu = (u - u.mean()) / max(u.std(), 1e-30)
        r = y - np.polyval(np.polyfit(uu, y, deg), uu)
        return math.sqrt(float(np.sum(r ** 2)) / max(y.size - deg - 1, 1)) / \
            max(float(np.std(y, ddof=1)), 1e-30)
    best = at(DEGREES[0])
    for d in DEGREES[1:]:
        v = at(d)
        if v < best * 0.95:
            best = v
        else:
            break
    return best


def rho_np(u, y):
    """Non-parametric cross-check of rho_poly (second differences along u); reported by
    run_benchmarks.py, not used by any verdict."""
    o = np.argsort(u); us, ys = u[o], y[o]
    dd = us[2:] - us[:-2]
    a = (us[2:] - us[1:-1]) / np.where(np.abs(dd) < 1e-30, 1e-30, dd)
    b = 1.0 - a
    e = a * ys[:-2] + b * ys[2:] - ys[1:-1]
    c2 = 1.0 / (a ** 2 + b ** 2 + 1.0)
    return math.sqrt(max(float(np.sum(c2 * e ** 2)) / max(us.size - 2, 1), 0.0)) / \
        max(float(np.std(ys, ddof=1)), 1e-30)


def window_rho(CX, CY, y, ab, sa, est, angle=None):
    """Best collapse residual over N_WIN directions in the window alpha_bar +/- K_WIN sigma_alpha
    / sqrt(M); returns (rho, the direction).  With `angle` given, evaluates `est` there."""
    if angle is not None:
        return est(math.cos(angle) * CX + math.sin(angle) * CY, y), angle
    half = K_WIN * math.radians(sa) / math.sqrt(M_ANCH)
    return min((est(math.cos(a) * CX + math.sin(a) * CY, y), a)
               for a in np.linspace(ab - half, ab + half, N_WIN))


def tau_min(pool, yn, n=None):
    """The resolution floor of stage two, eps_hat / (sqrt(n) s_y): eps_hat the standard deviation
    of the residual pool, s_y that of the averaged anchor responses (both with ddof = 1)."""
    n = N_MERGE if n is None else n
    return float(np.std(pool, ddof=1)) / (float(np.std(yn, ddof=1)) * math.sqrt(n))


def rho_true_hat(rho, tmin):
    """The structural residual: the floor taken off in quadrature, truncated at zero."""
    return math.sqrt(max(rho ** 2 - tmin ** 2, 0.0))


# ------------------------------------------------------------------ verdicts
# The statistic is compared with the tolerance by <= (a value equal to the declared tolerance
# is tolerated); the gate q95 / tau_min is compared by <, so an abstention can only come from
# the gate.  No run of the paper has a statistic exactly equal to its tolerance.
def v1(st, tau1, q95):
    if st > max(tau1, q95):
        return 'NOT-RED'
    if st <= tau1 and q95 < tau1:
        return 'REDUCIBLE'
    return 'UNRESOLVED'


def v2(rt, tmin):
    if rt > max(TAU, tmin):
        return 'not-red'
    if rt <= TAU and tmin < TAU:
        return 'reduce'
    return 'undet'


def joint(c1, c2):
    if c1 == 'REDUCIBLE' and c2 == 'reduce':
        return 'RED'
    if c1 == 'NOT-RED' or c2 == 'not-red':
        return 'NOT'
    return 'UND'


STAGE2_NAME = {"reduce": "REDUCIBLE", "not-red": "NOT-RED", "undet": "UNRESOLVED"}
JOINT_NAME = {"RED": "REDUCIBLE", "NOT": "NOT-RED", "UND": "UNRESOLVED"}
