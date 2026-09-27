# Two-stage reducibility framework — code and data

Code and data to reproduce every number, table and data-driven figure of

> Yan, D. Equal-response probing with replicate calibration identifies scaling relations from noisy observations of complex systems. Nature Communications (submitted).

The framework decides whether a two-input dimensionless response Π_o = f(Π_1, Π_2), evaluated
under additive noise of constant variance, reduces to a single group. Stage one tests whether
the level-set direction is constant (statistic σ̂_true against the tolerance τ₁ and the
simulated gate q₉₅); stage two tests whether the response collapses onto the group (statistic
ρ̂_true against τ and the floor τ_min). Verdicts are REDUCIBLE, NOT-RED or UNRESOLVED.

Code: MIT licence (`LICENSE`). Data and Source Data: CC BY 4.0 (`LICENSE-DATA`).
Repository: https://github.com/yandingbang/reducibility-two-input

## Layout

```
framework/          the method, one definition of everything
  core.py           protocol constants, anchors and probes, read-out, stage one (null
                    simulation, sigma_noise, q95), stage two (collapse residual, window
                    search, tau_min), the verdict rules v1 / v2 / joint
  benchmarks.py     the eight benchmark regimes: response functions, windows, truth labels
  synthetic.py      the synthetic field families and their random draws
benchmarks/         the main-text results
  fig2_data.py      one realisation of the protocol per regime  -> data/fig2_method_data.npz
  run_benchmarks.py 8 realisations per regime, verdict modes     -> data/framework_pipeline_v4.npz
  counterexample_directions.py       Supplementary Note 1        -> data/sigma_alpha_counterexample.npz
  counterexample_active_subspace.py  Supplementary Note 1, the active-subspace matrix (prints)
si_data/            the Supplementary figures and tables (data scripts, outputs in data/)
figures/            the figure scripts (common.py: fonts, colours, stage two recomputed)
source_data/        Source Data: export_source_data.py writes one CSV per panel (csv/), and
                    build_xlsx.py assembles them into Source_Data_<figure>.xlsx (needs openpyxl)
```

All paths are relative; run every script from any working directory with
`python path/to/script.py`.

## Environment

Python 3.8; numpy 1.24.3, scipy 1.10.1, matplotlib 3.5.0, torch 1.12.1 (CPU; only its random
number generator is used, for the anchor positions), pillow 10.4.0. `requirements.txt` pins
these versions. Figures use Arial and Times New Roman (installed system fonts).

## What produces what

| Script | Output | Used by | Time |
|---|---|---|---|
| `benchmarks/fig2_data.py` | `benchmarks/data/fig2_method_data.npz` | Table 1; Figs. 1b, 2d–e, 3b–c, 4, 5; Supplementary Table 4 | 4 min |
| `benchmarks/run_benchmarks.py` | `benchmarks/data/framework_pipeline_v4.npz` | verdict modes quoted by Fig. 5 | 4 min |
| `benchmarks/counterexample_directions.py` | `benchmarks/data/sigma_alpha_counterexample.npz` | Supplementary Note 1 | seconds |
| `benchmarks/counterexample_active_subspace.py` | prints | Supplementary Note 1 | seconds |
| `si_data/s1_resolution_linear.py` | `s1_linear_{0.25,1,4}.npz` | Supplementary Fig. 1 | 15 min per file |
| `si_data/s1_resolution_tworidge.py` | `s1_tworidge_{1,4}.npz` | Supplementary Fig. 1 | 15 min per file |
| `si_data/s2_null_calibration.py` | `s2_null_calib_600.npz` | Supplementary Fig. 2a | 15 min |
| `si_data/s2_null_fields.py` (+ `_summary.py`) | `s2_null_fields_800.npz` | Supplementary Fig. 2b | 40 min |
| `si_data/s3_floor_calibration.py` | `s3_floor_800_200.npz` | Supplementary Fig. 3 | 20 min |
| `si_data/s3_window_check.py`, `s3_degree_check.py` | `s3_window_check_800.npz`, `s3_degree_check_800.npz` | Supplementary Note 2 | 20 min |
| `si_data/s4_borderline.py` | `s4_borderline.npz` | Supplementary Fig. 4 | seconds |
| `si_data/s5_tau1_sweep.py` (+ `s5_tau1_summary.py`) | `s5_tau1_1200.npz` | Supplementary Fig. 5, Table 3 | 1 h |
| `si_data/s5_readout_check.py`, `s5_readout_sources.py` | `s5_readout_check.npz`, `s5_readout_sources.npz` | Supplementary Fig. 5 caption, Note 3 | 10 min each |
| `si_data/s6_probe_scale.py` | `s6_probe_scale.npz` | Supplementary Fig. 6 | 5 min |
| `si_data/s7_kappa.py` | prints | Supplementary Table 4, Note 3 | 7 min |
| `figures/fig1/fig1b_null.py` | `fig1b_null.png/.svg` | Fig. 1b (null distribution) | |
| `figures/fig2/fig2_pipe.py` (`PANELS=d,e`) | `fig2d_stage1`, `fig2e_stage2` | Fig. 2d, e | |
| `figures/fig3/plot_a_flow.py` | `fig3_a_flow` | Fig. 3a (micrographs in `fig3/img/`) | |
| `figures/fig3/plot_regions_field_hist.py`, `plot_collapse.py` | `fig3_stage1`, `fig3_stage2` | Fig. 3b, c | |
| `figures/fig4/plot_grid_single.py` | `fig4` | Fig. 4 | |
| `figures/fig5/plot_decision_map.py` | `fig5` (180 mm) | Fig. 5 (pictures in `fig5/img/`) | |
| `figures/si/plot_s*.py` | `figS1` … `figS6` | Supplementary Figs. 1–6 | |
| `source_data/export_source_data.py`, `build_xlsx.py` | `source_data/csv/`, `Source_Data_*.xlsx` | the values plotted in every data-driven panel | 1 min |

Times are for one CPU core. Every data file the figures need is included, so the figure scripts
run without regenerating anything; the data scripts regenerate the files bit for bit from the
seeds recorded in them (every regenerated file was compared with the included one).

The schematic panels of Figs. 1d and 2a–c carry no data and are not generated here; Fig. 3a
(`figures/fig3/plot_a_flow.py`) is a schematic around two micrographs. The
Fig. 1 schematics are included for completeness: `figures/fig1/fig1a_universe.py` (panel a,
the field with its noise band), `fig1b_spread.py` (panel b, the fan and the triangle; the
triangle's proportions are the slow-migration values of Table 1, typed in) and `fig1c_flow_v2.py`
(panel c, the flow chart); none of them reads data.

## Protocol constants and seeds

The protocol constants are in `framework/core.py` (M = 40 anchors, margin 0.20, K = 15 queried
angles t_j = jπ/14, n = 6 repeats, N_MC = 1200, N_F = 6400, degree ladder 2–20, window
ᾱ ± 3σ_α/√M with 21 candidates, τ₁ = 4°, τ = 5 %). Benchmark regime *i* (order of
`framework.benchmarks.UNITS`) uses the seed 20260918 + *i* for its anchors, seed + 7777 for
the 200 calibration anchors, and the noise generator `default_rng(13 · seed)` whose first
draw is the realisation of the main text. Synthetic field *j* of a design uses the seed
s₀ + *j* with s₀ given in each script's header; the derived streams (second ridge, λ, anchors,
calibration, observation noise, null simulation) are documented in `framework/synthetic.py`
and in the scripts.

Evaluated points. `framework.core.merged_run` draws errors only for the points a run of the
protocol evaluates, the K = 15 queried angles and the anchor, n times each: the residual pool
holds the 40 × 16 × 6 = 3840 within-repeat residuals of the run, and the profile is defined
only at the queried angles (nan elsewhere). The fine grid of 99 angles is used only by the
noise-free read-out study of Supplementary Fig. 6.

## Contents of the data files

All files are NumPy archives (`numpy.load(path, allow_pickle=True)`). Angles are in degrees
unless a key says radians; coordinates are natural logarithms of the dimensionless inputs;
`eps_mult` is the noise level as a multiple of the field's calibrated ε₀; verdict strings are
`REDUCIBLE / NOT-RED / UNRESOLVED` for stage one and the joint verdict, `reduce / not-red /
undet` for stage two.

**`benchmarks/data/fig2_method_data.npz`** — one realisation per regime, keys `u{i}_…`,
i = 0…7 in the order of `framework.benchmarks.UNITS`:

| key | shape | meaning |
|---|---|---|
| `name`, `reducible`, `box` | –, –, 4 | regime, truth label, window (x0, x1, y0, y1) |
| `CX`, `CY`, `HX`, `HY` | 40 | anchor coordinates and probe semi-axes |
| `t_all`, `t_q`, `idx_q` | 99, 15, 15 | fine-grid angles (rad), the queried angles and their indices |
| `prof_true`, `prof_mean`, `prof_reps` | 40×99, 40×99, 6×40×99 | signed profile: noise-free, averaged over the 6 repeats, each repeat |
| `cross_t` | 40 | interpolated zero crossing of the averaged profile (rad) |
| `ang`, `abar`, `sa` | 40, –, – | directions read (rad, mod π), axial mean (rad), σ_α |
| `sn`, `q95`, `st`, `verdict` | – | σ_noise, q₉₅, σ̂_true, stage-one verdict |
| `sa_null`, `st_null` | 6400 | null draws of σ_α and σ̂_true |
| `yn`, `s_y`, `eps`, `eps_pool` | 40, –, –, – | averaged anchor responses, their sd, ε, ε̂ from the residual pool |
| `pick`, `R_over_eps`, `n_signchg` | –, 40, 40 | anchor shown in Fig. 2b, profile amplitude / ε, sign changes per anchor |
| `TAU1`, `TAU`, `M_ANCH`, `N_MERGE`, `N_MC`, `N_FIELDSIM` | – | protocol constants |

**`benchmarks/data/framework_pipeline_v4.npz`** — 8 realisations per regime (`n_rep`), arrays of
length 8 in the order of `units`: `reducible`, `eps`, `s_y`, `sigma_alpha` and `sigma_alpha_sd`
(median and sd over realisations), `floor` (σ_noise), `q95`, `sigma_true` (σ̂_true), `rho_poly`,
`rho_np`, `tau_min`, `rho_true` (ρ̂_true; all medians), `stage1`, `stage2`, `joint` (verdict
modes), `joint_frac` (share of realisations giving the mode); `tau1`, `tau`, `M`, `frac_min`.

**`benchmarks/data/sigma_alpha_counterexample.npz`** — Supplementary Note 1: `n` (1, 4, 16),
`amp`, lattice `CX`, `CY`, `sigma_pointwise`, `alpha_bar_deg`, `rho` and `rho_dir_deg` (global
minimum and its direction), `rho_window`, `rho_window_dir_deg`, `rho_w2`, `s_y`.

**`si_data/data/`** — `rows` arrays carry a `cols` key naming the columns; `profile` gives the
ridge profile of each run; scalar keys record the settings.

| file | rows / keys | columns or keys |
|---|---|---|
| `s1_linear_{0.25,1,4}.npz`, `s1_tworidge_{1,4}.npz` | `rows` 800×8 (100 fields × 8 λ) | eps_mult, lam, field, sigma_grad (analytic σ_true), sigma_noise, q95, sigma_true_hat, detected (σ̂_true > q₉₅) |
| `s2_null_calib_600.npz` | 600 runs | `sn`, `q95`, `sa`, `st`, `exceed` (σ̂_true > q₉₅), `p` (share of the run's null draws ≥ its σ̂_true); `eps`, `seed` |
| `s2_null_fields_800.npz` | `rows` 1600×10 | field, eps_mult, eps, sn, q95, sa, st, exceed, p, floor (noise-free σ_α of the field) |
| `s3_floor_800_200.npz` | `rows` 4400×13 | kind (0 null, 1 tilted), field, lam, eps_mult, eps, s_y, rho, tau_min, rho_true (ρ̂_true), verdict (0 reduce, 1 not-red, 2 undet; `verdict_codes`), rho_clean, sigma_alpha, dir_err (deg) |
| `s3_window_check_800.npz` | `rows` 2400×12 | field, eps_mult, tau_min, rho_dep (deployed window), rho_floorwin (half-width floored at `w_min_deg`), rho_truth (at the true direction), rho_truth_clean (noise-free, true direction), e1_deg (stage-one direction error), half_deg (window half-width), err_dep_deg, err_floorwin_deg, sigma_alpha |
| `s3_degree_check_800.npz` | 2400 runs | `field`, `eps_mult`, `tau_min`, `r_noisy` and `r_clean` (2400×7: ρ at each degree of `degrees`, noisy and noise-free responses, true direction), `stop_noisy`, `stop_clean` (index where the ladder stops), `ratio_deployed_ladder` (ρ/τ_min) |
| `s4_borderline.npz` | 6 panels k = 0…5 | `lams`; per panel `u{k}` (ln û at 40 anchors), `y{k}` (averaged responses), `p{k}` (polynomial coefficients, highest first, on standardised u), `deg{k}`, `rho{k}`, `tau_min{k}`, `rho_true{k}`, `verdict{k}`; `profile`, `seed`, `tau` |
| `s5_tau1_1200.npz` | `rows` 2400×8 (1200 fields × 2 levels) | field, lam, eps_mult, sigma_true (analytic σ_grad), sigma_noise, q95, sigma_alpha, sigma_true_hat; `p_exact`, `lam_lo`, `lam_hi` |
| `s5_readout_check.npz` | 1200 fields | `field`, `lam`, `sigma_true`, `sa_clean` (noise-free σ_α at 15 angles), `sa_x1`, `sa_x4` (noisy σ_α) |
| `s5_readout_sources.npz` | 1200 fields | `sa_clean` 1200×6: noise-free σ_α under the six read-outs named in `configs` / `short` (15 or 99 angles; deployed probes or probes capped at 0.4, 0.1, 0.02); `sigma_true`, `lam` |
| `s6_probe_scale.npz` | 8 benchmarks × 6 probe fractions | `names`, `red`, `lams` (probe fractions), `truth` 8×2 (σ_grad by central differences with steps `h_truth`), `a_true` 8×40 (gradient directions, rad), `ang15`, `ang90` 8×6×40 (directions read at 15 angles / on the 99-angle grid, rad), `sig15`, `sig90` 8×6 (their σ_α), `cx`, `cy`, `hx`, `hy` (anchors and deployed probes) |
