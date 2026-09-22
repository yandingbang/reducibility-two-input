"""
counterexample_active_subspace.py --- how far the impossibility actually reaches.

sigma_alpha_counterexample.py built three fields

    y_n(w1, w2) = w2 + (A / 2 pi n) sin(2 pi n w1),   A = 1.481,  n in {1, 4, 16}

whose direction multisets, sampled at w1 = i/49, are identical element by element
(because gcd(n, 49) = 1 makes i -> n i mod 49 a bijection on Z_49), while the
collapse residual rho falls 0.316 -> 0.142 -> 0.036.  That refutes any grading
rule built on a statistic of the DIRECTION multiset.

An obvious objection: active subspaces does not use directions.  It uses the
uncentred second moment of the UNNORMALISED gradient,

    C = E[ grad y grad y^T ],

and the magnitude |grad y| carries information a unit vector throws away.  So the
published impossibility, as stated, leaves active subspaces untouched.

This script checks whether it should.  The gradient is

    grad y_n = ( A cos(2 pi n w1),  1 ),

so the 1/n in the amplitude is cancelled exactly by the n from differentiating.
The gradient VECTOR multiset -- not just its normalisation -- is therefore the
same for every n coprime to 49, and C with it.

Two versions are reported.

  (a) the 49-anchor lattice of Supplementary Note 1;
  (b) the continuum, w1 ~ Uniform[0, 1], where E[cos(2 pi n w1)] = 0 and
      E[cos^2(2 pi n w1)] = 1/2 for EVERY n >= 1, so C = diag(A^2/2, 1)
      identically -- no number theory needed, and no restriction on n.

Output: printed table.
"""
import math
import numpy as np

A = 1.481
P = 49
NS = (1, 4, 16)
RHO = {1: 0.316, 4: 0.142, 16: 0.036}      # measured in sigma_alpha_counterexample.py


def grad(n, w1):
    return np.stack([A * np.cos(2 * math.pi * n * w1), np.ones_like(w1)], axis=1)


def report(tag, w1, wt=None):
    """wt = quadrature weights summing to 1.  None means the plain sample mean,
    which is the right thing for a lattice of equally likely anchors."""
    w = np.full(w1.size, 1.0 / w1.size) if wt is None else wt / wt.sum()
    print(f'\n{tag}   ({w1.size} samples)')
    print(f'  {"n":>3}{"C11":>16}{"C12":>16}{"C22":>16}'
          f'{"lam1":>12}{"lam2":>12}{"gap":>10}{"axis/deg":>11}{"rho":>9}')
    Cs = []
    for n in NS:
        g = grad(n, w1)
        C = (g * w[:, None]).T @ g
        lam, V = np.linalg.eigh(C)
        v = V[:, -1]
        Cs.append(C)
        ax = math.degrees(math.atan2(v[1], v[0])) % 180.0
        print(f'  {n:>3}{C[0, 0]:16.12f}{C[0, 1]:16.12f}{C[1, 1]:16.12f}'
              f'{lam[-1]:12.8f}{lam[0]:12.8f}{lam[-1] / lam[0]:10.6f}'
              f'{min(ax, 180.0 - ax):11.6f}{RHO[n]:9.3f}')
    d = max(float(np.abs(Cs[0] - c).max()) for c in Cs[1:])
    print(f'  max |C_1 - C_n| over the family : {d:.3e}')
    return d


print('=' * 104)
print('  Does the counterexample also defeat active subspaces?')
print('=' * 104)

d_lat = report('(a) 49-anchor lattice, w1 = i/49', np.arange(P) / P)

# NOTE: Gauss-Legendre nodes are NOT uniformly spaced -- they cluster at the ends.
# The weights are what make the node set represent Lebesgue measure, so they must
# be passed in.  Averaging over the bare nodes gives a different (and n-dependent)
# matrix, which is an artefact of the quadrature, not a property of the field.
_x, _w = np.polynomial.legendre.leggauss(4096)
d_con = report('(b) continuum, w1 ~ U[0,1] (Gauss-Legendre, 4096 nodes + weights)',
               (_x + 1) / 2, _w)

print('\n' + '=' * 104)
print(f'  rho spans {RHO[1] / RHO[16]:.1f}x across the family (0.316 -> 0.036).')
print(f'  C is constant to {max(d_lat, d_con):.1e} -- so its eigenvalues, its spectral gap and')
print('  its dominant eigenvector are constant too.')
print()
print('  Therefore the impossibility is NOT confined to statistics of the direction')
print('  multiset.  It covers any functional of the gradient-vector multiset, and the')
print('  active-subspace matrix C = E[grad y grad y^T] is one of those.  On this family')
print('  active subspaces returns one answer while the true collapse residual moves by')
print('  nearly an order of magnitude.')
print()
print('  In the continuum the restriction gcd(n, 49) = 1 is not needed either: C is the')
print('  same for EVERY n >= 1, because w1 -> n w1 mod 1 preserves Lebesgue measure.')
print('  The lattice version exists so that a run with 49 anchors reproduces')
print('  it exactly rather than up to sampling error.')
print()
print('  Sharper still, and worth stating carefully.  C is nearly isotropic here: the')
print('  eigenvalue gap is 1.097, which is the standard diagnostic saying "no adequate')
print('  one-dimensional active subspace".  That reading is right for n = 1, where')
print('  rho = 0.316, and WRONG for n = 16, where the field collapses to within 3.6%.')
print('  And the dominant eigenvector points along w1, while the variable the n = 16')
print('  field actually collapses onto is w2 -- the amplitude A / 2 pi n shrinks with n')
print('  but the gradient amplitude A does not, so the axis that carries the most')
print('  gradient energy is not the axis that carries the structure.')
print()
print('  What this does NOT say: active subspaces is not being refuted as a method.  It')
print('  answers the question it was built for -- "along which direction does y vary')
print('  most" -- and it answers that consistently.  The point is narrower and it is')
print('  exactly F2: no functional of the gradient sample can GRADE how well a collapse')
print('  holds, because grading needs the spatial arrangement such a functional throws')
print('  away.  A fast oscillation and a slow drift can share every gradient statistic')
print('  and still differ by an order of magnitude in whether the collapse survives.')
