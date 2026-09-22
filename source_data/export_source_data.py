# -*- coding: utf-8 -*-
"""Source Data -- the numbers behind every data-driven panel, one CSV per panel.

Each CSV is exactly what the corresponding figure script draws, computed with the same code
(figures/common.py for stage two, the same binning and intervals as the plot scripts).  The
CSVs go to csv/<figure>/<panel>.csv; build_xlsx.py assembles them into one workbook per
figure (Source_Data_Fig1.xlsx, ..., Source_Data_SuppFig6.xlsx), one sheet per panel.

Angles in degrees, coordinates in natural logarithms of the dimensionless inputs.
"""
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "figures"))
sys.path.insert(0, os.path.join(HERE, ".."))
import common as C                                                # noqa: E402
from framework.core import TAU, TAU1, v1                          # noqa: E402

SI = os.path.join(HERE, "..", "si_data", "data")
OUT = os.path.join(HERE, "csv")


def write(figure, panel, header, rows, notes=()):
    """one CSV: note lines first (prefixed '#'), then the header and the rows"""
    d = os.path.join(OUT, figure)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, panel + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for n in notes:
            w.writerow(["# " + n])
        w.writerow(header)
        for r in rows:
            w.writerow([("%.10g" % v) if isinstance(v, (float, np.floating)) else v for v in r])
    print("  %-14s %-28s %5d rows" % (figure, panel, len(rows)))


def cp(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return 100 * lo, 100 * hi


def unwrap_deg(ang_rad, centre_deg):
    a = np.degrees(np.asarray(ang_rad))
    return centre_deg + ((a - centre_deg + 90.0) % 180.0) - 90.0


d = C.load()
S = {"u%d" % i: C.stages(d, "u%d" % i) for i in range(8)}
STAGE1 = {"REDUCIBLE": "sigma_true_hat <= tau_1 and q95 < tau_1", "NOT-RED": "sigma_true_hat > max(tau_1, q95)",
          "UNRESOLVED": "neither"}
STAGE2 = {"reduce": "rho_true_hat <= tau and tau_min < tau", "not-red": "rho_true_hat > max(tau, tau_min)",
          "undet": "neither"}


# ---------------------------------------------------------------- Fig. 1b: the null distribution
def fig1b():
    u = "u0"
    null = np.asarray(d[u + "_st_null"], float)
    q95 = float(d[u + "_q95"])
    NB = 16
    N = null.size
    tail = null[null > 0.0]
    XMAX = float(np.percentile(null, 99.5)) * 1.15
    edges = np.linspace(0.0, XMAX, NB + 1)
    counts, _ = np.histogram(tail, bins=edges)
    notes = ["Fig. 1b: null distribution of sigma_true_hat for %s (%d synthetic runs of the null simulation)" % (str(d[u + "_name"]), N),
             "q95 = %.6g deg; share exactly zero = %.4f; bins cover the non-zero values, fraction is of all %d samples" % (q95, np.mean(null == 0), N)]
    write("Fig1", "Fig1b_histogram", ["bin_lower_deg", "bin_upper_deg", "count", "fraction_of_all"],
          [(edges[i], edges[i + 1], int(counts[i]), counts[i] / N) for i in range(NB)], notes)
    write("Fig1", "Fig1b_null_samples", ["sigma_true_hat_null_deg"], [(v,) for v in null],
          ["the %d null draws of sigma_true_hat, %s" % (N, str(d[u + "_name"]))])


# ---------------------------------------------------------------- stage-one and stage-two panels
def stage1_panel(figure, panel, units, xname, yname, note):
    rows, scal = [], []
    for u in units:
        s = S[u]
        centre = math.degrees(s["abar"])
        ang = unwrap_deg(s["ang"], centre)
        for i in range(len(ang)):
            rows.append((s["name"], i, s["CX"][i], s["CY"][i], ang[i]))
        scal.append((s["name"], centre % 180.0, s["sa"], s["sn"], s["q95"], s["st"], TAU1, s["c1"], STAGE1[s["c1"]]))
    write(figure, panel + "_directions", ["regime", "anchor", xname, yname, "direction_deg"], rows,
          [note, "direction_deg: the direction read at the anchor, axial (mod 180), unwrapped to within 90 deg of the regime mean;",
           "the density curves are Gaussian kernel density estimates of these values with bandwidth 2.5 deg"])
    write(figure, panel + "_stage_one", ["regime", "alpha_bar_deg", "sigma_alpha_deg", "sigma_noise_deg", "q95_deg",
                                         "sigma_true_hat_deg", "tau_1_deg", "stage_one", "condition"], scal, [note])


def stage2_panel(figure, panel, units, yname, note):
    rows, scal = [], []
    for u in units:
        s = S[u]
        x0, x1, y0, y1 = s["box"]
        uu, yy = s["u"], s["yn"]
        m, sd = uu.mean(), uu.std()
        coef = np.polyfit((uu - m) / sd, yy, s["deg"])
        for i in range(len(uu)):
            rows.append((s["name"], i, uu[i], yy[i], (s["CX"][i] - x0) / (x1 - x0), (s["CY"][i] - y0) / (y1 - y0)))
        a = math.degrees(s["a_star"]) % 180.0
        scal.append((s["name"], a, math.cos(s["a_star"]), math.sin(s["a_star"]), s["deg"], m, sd,
                     " ".join("%.10g" % c for c in coef), s["rho"], s["tmin"], s["rt"], TAU,
                     C.STAGE2_NAME[s["c2"]], STAGE2[s["c2"]], C.JOINT_NAME[s["jt"]]))
    write(figure, panel + "_collapse", ["regime", "anchor", "ln_u_hat", yname, "position_x_fraction", "position_y_fraction"], rows,
          [note, "ln_u_hat = cos(a*) ln Pi_1 + sin(a*) ln Pi_2 at the searched direction a*; position fractions place the anchor in its window (point colour)"])
    write(figure, panel + "_stage_two", ["regime", "a_star_deg", "exponent_Pi1", "exponent_Pi2", "polynomial_degree", "u_mean", "u_sd",
                                         "polynomial_coefficients_highest_first_on_(u-u_mean)/u_sd", "rho", "tau_min", "rho_true_hat",
                                         "tau", "stage_two", "condition", "joint_verdict"], scal, [note])


# ---------------------------------------------------------------- Fig. 4: the two exact reductions
def fig4():
    units = ("u2", "u3")
    exact = {"u2": (-1.0, 1.0 / 3.0), "u3": (2.0, 1.0)}
    rows, scal = [], []
    for u in units:
        s = S[u]
        centre = math.degrees(s["abar"])
        ang = unwrap_deg(s["ang"], centre)
        ea, eb = exact[u]
        exact_deg = math.degrees(math.atan2(eb, ea)) % 180.0
        for i in range(len(ang)):
            rows.append((s["name"], i, s["CX"][i], s["CY"][i], ang[i]))
        scal.append((s["name"], centre % 180.0, exact_deg, s["sa"], s["sn"], s["q95"], s["st"], TAU1, s["c1"], STAGE1[s["c1"]]))
    write("Fig4", "Fig4_directions", ["regime", "anchor", "ln_Pi1", "ln_Pi2", "direction_deg"], rows,
          ["Fig. 4, stage-one row: microfluidic (ln Ca, ln phi) and column buckling (ln(kappa L/r), ln(sigma_y/E))",
           "direction_deg unwrapped to within 90 deg of the regime mean; density curves: Gaussian KDE, bandwidth 2.5 deg"])
    write("Fig4", "Fig4_stage_one", ["regime", "alpha_bar_deg", "exact_direction_deg", "sigma_alpha_deg", "sigma_noise_deg", "q95_deg",
                                     "sigma_true_hat_deg", "tau_1_deg", "stage_one", "condition"], scal,
          ["exact_direction_deg: the direction of the known group, Ca^-1 phi^(1/3) and (kappa L/r)^2 sigma_y/E"])
    stage2_panel("Fig4", "Fig4", units, "response", "Fig. 4, stage-two row: response = L_d/w (microfluidic), sigma_cr/sigma_y (buckling)")


# ---------------------------------------------------------------- Fig. 5: the decision map
def fig5():
    p = np.load(C.PUBLISHED, allow_pickle=True)
    modes = {str(u): C.JOINT_NAME[str(j)] for u, j in zip(p["units"], p["joint"])}
    rows = []
    for i in range(8):
        s = S["u%d" % i]
        rows.append((s["name"], s["st"] / max(TAU1, s["q95"]), s["rt"] / max(TAU, s["tmin"]), s["st"], s["q95"], TAU1,
                     s["rt"], s["tmin"], TAU, C.JOINT_NAME[s["jt"]], modes[s["name"]]))
    write("Fig5", "Fig5_decision_map", ["regime", "x_stage_one_ratio", "y_stage_two_ratio", "sigma_true_hat_deg", "q95_deg", "tau_1_deg",
                                        "rho_true_hat", "tau_min", "tau", "verdict", "verdict_mode_over_8_realisations"], rows,
          ["Fig. 5: x = sigma_true_hat / max(tau_1, q95), y = rho_true_hat / max(tau, tau_min); the realisation of Table 1",
           "the last column is the mode of the joint verdict over the 8 realisations of benchmarks/run_benchmarks.py"])


# ---------------------------------------------------------------- Supplementary Fig. 1
def s1():
    EDG = np.concatenate([np.geomspace(0.05, 4.0, 13), [8.0, np.inf]])
    curves, raw = [], []
    for fam, pattern in (("single ridge + linear term", "s1_linear_%g.npz"), ("two ridges", "s1_tworidge_%g.npz")):
        for m in (1.0, 4.0):
            A = np.load(os.path.join(SI, pattern % m))["rows"]
            x, det = A[:, 3] / np.maximum(A[:, 5], 1e-12), A[:, 7]
            for r in A:
                raw.append((fam, m, r[1], int(r[2]), r[3], r[5], r[6], r[3] / max(r[5], 1e-12), int(r[7])))
            for lo, hi in zip(EDG[:-1], EDG[1:]):
                k = (x >= lo) & (x < hi)
                if k.sum() >= 8:
                    curves.append((fam, m, lo, hi, int(k.sum()), float(np.median(x[k])), 100.0 * float(det[k].mean())))
    write("SuppFig1", "SuppFig1_curves", ["family", "noise_multiple", "bin_lower", "bin_upper", "n_runs", "x_median", "detected_pct"], curves,
          ["Supplementary Fig. 1: detection rate (sigma_true_hat > q95) against sigma_grad / q95, 13 logarithmic bins on [0.05, 4] then [4, 8) and [8, inf);",
           "a bin is drawn when it holds at least 8 runs; the point sits at the median x of the bin; lam = 0 runs (sigma_grad = 0) are not drawn"])
    write("SuppFig1", "SuppFig1_runs", ["family", "noise_multiple", "lam", "field", "sigma_grad_deg", "q95_deg", "sigma_true_hat_deg", "x", "detected"], raw,
          ["every run behind the curves: x = sigma_grad / q95, detected = 1 if sigma_true_hat > q95"])


# ---------------------------------------------------------------- Supplementary Fig. 2
def s2():
    one_p = np.load(os.path.join(SI, "s2_null_calib_600.npz"))["p"]
    f = np.load(os.path.join(SI, "s2_null_fields_800.npz"), allow_pickle=True)
    A, c = f["rows"], {str(k): i for i, k in enumerate(f["cols"])}
    alpha = np.linspace(0.0, 0.20, 2001)
    curves = [("one field, 600 runs", one_p)] + [("800 fields, noise x%g" % m, A[A[:, c["eps_mult"]] == m, c["p"]]) for m in (1.0, 4.0)]
    rows = [(lab, 100 * a, 100 * float(np.mean(p <= a))) for lab, p in curves for a in alpha]
    write("SuppFig2", "SuppFig2a_curves", ["series", "nominal_rate_pct", "observed_rate_pct"], rows,
          ["Supplementary Fig. 2a: share of runs whose p (share of the run's own null draws reaching its sigma_true_hat) is at or below the nominal rate;",
           "the shaded band is the 95% binomial interval of the one-field series, 1.96 sqrt(a (1 - a) / 600)"])
    write("SuppFig2", "SuppFig2a_p_values", ["series", "run_or_field", "noise_multiple", "p"],
          [("one field", i, 1.0, v) for i, v in enumerate(one_p)] +
          [("800 fields", int(r[c["field"]]), r[c["eps_mult"]], r[c["p"]]) for r in A])
    rows = []
    for m in (1.0, 4.0):
        D = A[A[:, c["eps_mult"]] == m]
        ratio = D[:, c["q95"]] / np.maximum(D[:, c["floor"]], 1e-12)
        for g, lo, hi in (("all fields", None, None), ("< 3", 0.0, 3.0), ("3-10", 3.0, 10.0), (">= 10", 10.0, np.inf)):
            sel = np.ones(len(D), bool) if lo is None else (ratio >= lo) & (ratio < hi)
            n, k = int(sel.sum()), int(D[sel, c["exceed"]].sum())
            l_, h_ = cp(k, n) if n else (float("nan"), float("nan"))
            rows.append((m, g, k, n, 100.0 * k / n if n else float("nan"), l_, h_))
    write("SuppFig2", "SuppFig2b_rates", ["noise_multiple", "group_q95_over_readout_floor", "runs_above_q95", "n_runs", "rate_pct", "ci95_low_pct", "ci95_high_pct"], rows,
          ["Supplementary Fig. 2b: false-alarm rate at q95 on exactly reducible fields, Clopper-Pearson 95% intervals"])
    write("SuppFig2", "SuppFig2b_runs", ["field", "noise_multiple", "q95_deg", "readout_floor_deg", "sigma_true_hat_deg", "above_q95"],
          [(int(r[c["field"]]), r[c["eps_mult"]], r[c["q95"]], r[c["floor"]], r[c["st"]], int(r[c["exceed"]])) for r in A])


# ---------------------------------------------------------------- Supplementary Fig. 3
def s3():
    f = np.load(os.path.join(SI, "s3_floor_800_200.npz"), allow_pickle=True)
    A, c = f["rows"], {str(k): i for i, k in enumerate(f["cols"])}
    VN = {0.0: "REDUCIBLE", 1.0: "NOT-RED", 2.0: "UNRESOLVED"}
    null = A[A[:, c["kind"]] == 0]
    write("SuppFig3", "SuppFig3a", ["noise_multiple", "field", "tau_min", "rho"],
          [(r[c["eps_mult"]], int(r[c["field"]]), r[c["tau_min"]], r[c["rho"]]) for r in null],
          ["Supplementary Fig. 3a: exactly reducible fields, collapse residual rho against the floor tau_min"])
    tilt = A[A[:, c["kind"]] == 1]
    write("SuppFig3", "SuppFig3b", ["noise_multiple", "lam", "field", "rho_clean", "rho_true_hat", "tau_min", "stage_two"],
          [(r[c["eps_mult"]], r[c["lam"]], int(r[c["field"]]), r[c["rho_clean"]], r[c["rho_true"]], r[c["tau_min"]], VN[r[c["verdict"]]]) for r in tilt],
          ["Supplementary Fig. 3b: fields with a departure, rho_true_hat against the noise-free residual rho_clean; tau = %.2f" % float(f["tau"])])


# ---------------------------------------------------------------- Supplementary Fig. 4
def s4():
    f = np.load(os.path.join(SI, "s4_borderline.npz"), allow_pickle=True)
    VN = {"reduce": "REDUCIBLE", "not-red": "NOT-RED", "undet": "UNRESOLVED"}
    rows, scal = [], []
    for k, lam in enumerate(f["lams"]):
        u, y, p = f["u%d" % k], f["y%d" % k], f["p%d" % k]
        for i in range(len(u)):
            rows.append((lam, i, u[i], y[i]))
        scal.append((lam, int(f["deg%d" % k]), u.mean(), u.std(), " ".join("%.10g" % c for c in p), float(f["rho%d" % k]),
                     float(f["tau_min%d" % k]), float(f["rho_true%d" % k]), float(f["tau"]), VN[str(f["verdict%d" % k])]))
    write("SuppFig4", "SuppFig4_points", ["lam", "anchor", "ln_u_hat", "response"], rows,
          ["Supplementary Fig. 4: one synthetic field (seed %d, profile %s) at six values of the transverse term lam" % (int(f["seed"]), str(f["profile"]))])
    write("SuppFig4", "SuppFig4_fits", ["lam", "polynomial_degree", "u_mean", "u_sd", "polynomial_coefficients_highest_first_on_(u-u_mean)/u_sd",
                                        "rho", "tau_min", "rho_true_hat", "tau", "stage_two"], scal)


# ---------------------------------------------------------------- Supplementary Fig. 5
def s5():
    f = np.load(os.path.join(SI, "s5_tau1_1200.npz"), allow_pickle=True)
    A, c = f["rows"], {str(k): i for i, k in enumerate(f["cols"])}
    TAUS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    ra, rb = [], []
    for m in (1.0, 4.0):
        D = A[A[:, c["eps_mult"]] == m]
        s, q, st = D[:, c["sigma_true"]], D[:, c["q95"]], D[:, c["sigma_true_hat"]]
        for t in TAUS:
            v = np.array([v1(a, t, b) for a, b in zip(st, q)])
            ra.append((m, t, int(np.sum(v == "UNRESOLVED")), len(D), 100.0 * np.mean(v == "UNRESOLVED")))
            for err, mask, pop in (("wrongly NOT-RED", v == "NOT-RED", s < t), ("wrongly REDUCIBLE", v == "REDUCIBLE", s > t)):
                k, n = int(np.sum(mask & pop)), int(pop.sum())
                lo, hi = cp(k, n)
                rb.append((m, err, t, k, n, 100.0 * k / n, lo, hi))
    write("SuppFig5", "SuppFig5a", ["noise_multiple", "tau_1_deg", "unresolved_runs", "n_runs", "unresolved_pct"], ra,
          ["Supplementary Fig. 5a: share of the 1200 runs per noise level returned UNRESOLVED, re-scored at each tau_1"])
    write("SuppFig5", "SuppFig5b", ["noise_multiple", "error", "tau_1_deg", "wrong_runs", "population", "rate_pct", "ci95_low_pct", "ci95_high_pct"], rb,
          ["Supplementary Fig. 5b: wrongly NOT-RED among runs with sigma_grad < tau_1; wrongly REDUCIBLE among runs with sigma_grad > tau_1; Clopper-Pearson 95% intervals"])
    write("SuppFig5", "SuppFig5_runs", ["field", "noise_multiple", "lam", "sigma_grad_deg", "sigma_noise_deg", "q95_deg", "sigma_alpha_deg", "sigma_true_hat_deg"],
          [(int(r[c["field"]]), r[c["eps_mult"]], r[c["lam"]], r[c["sigma_true"]], r[c["sigma_noise"]], r[c["q95"]], r[c["sigma_alpha"]], r[c["sigma_true_hat"]]) for r in A],
          ["every run behind panels a and b"])


# ---------------------------------------------------------------- Supplementary Fig. 6
def s6():
    f = np.load(os.path.join(SI, "s6_probe_scale.npz"), allow_pickle=True)
    names, lams = [str(n) for n in f["names"]], f["lams"]
    dwrap = lambda a: (np.degrees(a) + 90.0) % 180.0 - 90.0
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    ra, rb = [], []
    for u, n in enumerate(names):
        for l, lam in enumerate(lams):
            ra.append((n, lam, f["sig15"][u, l], f["truth"][u, 0], abs(f["sig15"][u, l] - f["truth"][u, 0])))
        rb.append((n, rms(dwrap(f["ang90"][u, 0] - f["a_true"][u])), rms(dwrap(f["ang15"][u, 0] - f["ang90"][u, 0]))))
    write("SuppFig6", "SuppFig6a", ["benchmark", "probe_fraction", "sigma_alpha_clean_15_angles_deg", "sigma_grad_deg", "readout_error_deg"], ra,
          ["Supplementary Fig. 6a: noise-free read-out error |sigma_alpha,clean - sigma_grad| at 15 angles against probe size (fraction of the deployed probe)"])
    write("SuppFig6", "SuppFig6b", ["benchmark", "chord_rms_deg", "interpolation_rms_deg"], rb,
          ["Supplementary Fig. 6b: per-anchor RMS read-out error at the deployed probe; chord = %d-angle read-out minus the gradient direction, interpolation = 15-angle minus %d-angle read-out;" % (int(f["n_all"]), int(f["n_all"])),
           "for low Re, microfluidic and buckling the contours are straight and the chord term is exactly 0 (the value listed is the interpolation left on the fine grid)"])


if __name__ == "__main__":
    print("Source Data CSVs -> %s" % OUT)
    fig1b()
    stage1_panel("Fig2", "Fig2d", ("u0", "u4", "u1"), "ln_Re", "ln_eps_over_D", "Fig. 2d: rough pipe, low / medium / high Re")
    stage2_panel("Fig2", "Fig2e", ("u0", "u4", "u1"), "lambda", "Fig. 2e: rough pipe, friction factor lambda against ln u_hat")
    stage1_panel("Fig3", "Fig3b", ("u7", "u6", "u5"), "ln_Pi1", "ln_Pi2", "Fig. 3b: wound closure, slow / intermediate / fast migration")
    stage2_panel("Fig3", "Fig3c", ("u7", "u6", "u5"), "T_c_r", "Fig. 3c: wound closure, closure time T_c r against ln u_hat")
    fig4()
    fig5()
    s1(); s2(); s3(); s4(); s5(); s6()
    print("done")
