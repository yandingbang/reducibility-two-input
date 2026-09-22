# -*- coding: utf-8 -*-
"""The eight benchmark regimes (Methods, "Benchmark systems"; Supplementary Table 1).

Every response is a function of the logarithms of the two dimensionless inputs, so that the
framework works in log coordinates: resp(ln Pi_1, ln Pi_2) = Pi_o.  Windows are (x0, x1, y0, y1)
in the same coordinates.
"""
import math

import numpy as np
from scipy.linalg import solve_banded

PI2 = math.pi ** 2


def resp_pipe(xw, yw):
    """Darcy friction factor, Colebrook relation solved by fixed-point iteration;
    xw = ln Re, yw = ln(eps/D)."""
    Re, epsD = math.exp(xw), math.exp(yw)
    f = 0.02
    for _ in range(80):
        f = 1.0 / (-2.0 * math.log10(epsD / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** 2
    return f


def resp_pipe_lam(xw, yw):
    """Laminar Darcy friction factor f = 64 / Re (low-Re regime); independent of eps/D."""
    return 64.0 / math.exp(xw)


def resp_micro(xw, yw):
    """Droplet length L_d / w = Ca^-1 phi^(1/3), a constructed power law; xw = ln Ca, yw = ln phi."""
    return math.exp(-xw + yw / 3.0)


def resp_buck(xw, yw):
    """Rankine-Gordon column buckling, sigma_cr / sigma_y = pi^2 / (pi^2 + (kappa L / r)^2 sigma_y / E);
    xw = ln(kappa L / r), yw = ln(sigma_y / E)."""
    return PI2 / (PI2 + math.exp(2.0 * xw + yw))


# Fisher-KPP wound closure: half-domain x in [0, X_MAX], NX points, no-flux boundaries,
# initial density 0 for x < 1/2 and Pi_2 otherwise; diffusion implicit, reaction explicit;
# the closure time is the first crossing of n(0, t) = THC, interpolated between steps.
_Xmax, _Nx, _dt, _THC, _TMAX = 5.0, 161, 0.03, 0.5, 120.0
_xg = np.linspace(0, _Xmax, _Nx); _dx = _xg[1] - _xg[0]


def _closure(Pi1, Pi2):
    n = np.where(_xg < 0.5, 0.0, Pi2).astype(float)
    a = _dt * Pi1 / _dx ** 2
    ab = np.zeros((3, _Nx)); ab[0, 1:] = -a; ab[2, :-1] = -a; ab[1, :] = 1 + 2 * a
    ab[0, 1] = -2 * a; ab[2, -2] = -2 * a
    t = 0.0
    while t < _TMAX:
        p0 = n[0]; n = solve_banded((1, 1), ab, n + _dt * n * (1 - n)); t += _dt
        if n[0] >= _THC:
            return t - _dt + (_THC - p0) / (n[0] - p0 + 1e-12) * _dt
    return _TMAX


def resp_bio(xw, yw):
    """Dimensionless wound closure time; xw = ln Pi_1, yw = ln Pi_2."""
    return _closure(math.exp(xw), math.exp(yw))


CASES = {
    'Rough-pipe': dict(resp=resp_pipe,
                       boxes=[(math.log(1e4), math.log(1e5), math.log(2e-3), math.log(2e-2))]),
    'Wound closure': dict(resp=resp_bio,
                          boxes=[(math.log(6e-4), math.log(5e-3), math.log(0.40), math.log(0.90)),
                                 (math.log(0.03), math.log(3.0),  math.log(0.12), math.log(0.90)),
                                 (math.log(8.0),  math.log(50.0), math.log(0.10), math.log(0.45))]),
    'Microfluidic': dict(resp=resp_micro,
                         boxes=[(math.log(1e-2), math.log(3e-1), math.log(1e-1), math.log(2.0))]),
    'Column buckling': dict(resp=resp_buck,
                            boxes=[(math.log(20.0), math.log(200.0), math.log(1e-3), math.log(1e-2))]),
    'Pipe low-Re': dict(resp=resp_pipe_lam,
                        boxes=[(math.log(1e2), math.log(2e3), math.log(1e-3), math.log(5e-2))]),
    'Pipe high-Re': dict(resp=resp_pipe,
                         boxes=[(math.log(5e6), math.log(5e7), math.log(2e-3), math.log(5e-2))]),
}

# (display name, CASES key, window index, reducible at the tolerances of the paper).
# The order is the unit index ui; a regime's seed is RSEED + ui.
UNITS = [
    ('Rough-pipe, low Re',    'Pipe low-Re',     None, True),
    ('Rough-pipe, high Re',   'Pipe high-Re',    None, True),
    ('Microfluidic',          'Microfluidic',    None, True),
    ('Column buckling',       'Column buckling', None, True),
    ('Rough-pipe, medium Re', 'Rough-pipe',      None, False),
    ('Wound, fast migration', 'Wound closure',   2,    True),
    ('Wound, intermediate',   'Wound closure',   1,    False),
    ('Wound, slow migration', 'Wound closure',   0,    True),
]
