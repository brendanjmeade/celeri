# Correctness and consistency audit of the `celeri-solve` path (September 2026)

Branch `vet`, based on `main` at `015836c`. The audit covered every module that
`celeri-solve` reaches: configuration and command-line handling, model building (block closure,
meshes, segment processing), the elastic and kinematic operators (Okada, cutde triangular
dislocations, block rotations, strain, Mogi), the four solvers (`dense`, `dense_no_meshes`,
`qp`, `qp2`, `mcmc`), output writing, and the output contract consumed by `celeri-forward`.
The MCMC path was reviewed at full depth; the other solvers received a consistency pass.

## 1. Method

1. **Code mapping.** Three passes mapped the call graph, unit and sign conventions, and test
   coverage, producing a catalogue of suspects (D1–D37 below).
2. **Numerical verification harness.** Nineteen checks (S1–S10) confirmed or refuted every
   suspect that concerned geometry, signs, or units, using the two shipped test models
   (`tests/configs/test_japan_config.json`, `tests/configs/test_wna_config.json`), the demo
   WNA model, and a synthetic three-block model built for the purpose. The harness lives in
   the session scratchpad (`s0_geometry.py` … `s9_solve_types.py`); the durable checks were
   promoted to tests (section 6).
3. **Four review lanes** (MCMC; operators and kernels; model, closure, mesh, config; solvers,
   output and forward) re-read the code with the harness results in hand and ran their own
   experiments. Their findings are labelled R1-*, R2-*, R3-*, R4-*.
4. **Fixes.** Every clear-cut defect was fixed in its own commit with a regression test that
   fails on `main`; numerics-changing geometry fixes were landed after the harness confirmed
   them (Brendan's decision). Design and science questions are listed as proposals.

### Conventions taken as the specification

Positive strike-slip is left-lateral, positive dip-slip is reverse, positive tensile slip is
extension; a segment with `dip < 90` has its hanging wall to the right of the strike direction
(endpoints ordered west-first) and `dip > 90` to the left; station velocity = block rotation −
elastic (Okada segments + TDEs) + block strain + Mogi; residual = model − observed; segment
locking depths in km positive-down; mesh depths in km negative-down; longitudes in [0, 360);
rotation vectors in milli-rad/yr with `rotation_to_velocities` in m/rad so that products are
mm/yr.

## 2. Headline results

| # | Result | Evidence |
|---|---|---|
| 1 | **Triangle strike and dip were biased by the missing cos(latitude) factor.** `Mesh.from_params` built element normals from (Δlon, Δlat, Δr) legs. On the shipped meshes strikes were off by up to 12.8° and dips by up to 8.9°; a synthetic 60°-dipping fault at 40° N loaded as 53°. Because the kinematic dip slip scales with 1/cos(dip), kinematic rates and every coupling ratio built from them were wrong by 5–20 % of the relative plate motion, and a fully coupled mesh left a velocity discontinuity of that size across its trace. **Fixed** (strike, dip and centroids now come from the Cartesian vertices in a local east-north-up frame; the vertex reordering that landed with it was reverted, see section 8). | S3a, S1c, S1d: Japan transects: median trace step 6–7 % of plate motion before, 0.4–0.9 % after; synthetic model: kinematic dip slip 33.1 → 40.0 mm/yr (segment value 39.9), TDE-vs-Okada mismatch 4.0 → 0.05 mm/yr. |
| 2 | **The block strain-rate operator's shear column was a rigid rotation, not shear**, and the operator changed sign structure south of the equator (colatitude convention plus a one-sided negation). Shear strain could not be modelled and the shear column was nearly collinear with the block's Euler vector. **Fixed** (strain applied as a symmetric tensor in a local east-north frame; both hemispheres). | R2-3, R3-2: unit ε_λφ gave ∂u_e/∂y = −1, ∂u_n/∂x = +1 (shear 0.000, rotation 0.999); Japan block `tottori` had strain estimated. New finite-difference test in both hemispheres. |
| 3 | **The Okada geographic-frame handling is correct.** The mapping pass suspected the commented-out alternative; the harness showed the live code (stations rotated into the fault frame with the projected strike, displacements rotated back with the true azimuth) matches an independent transverse-Mercator + cutde reference to ~1e-3 on short segments, and that the alternative would be wrong by 2·sin(δ/2) with δ the oblique-Mercator grid convergence (median 8°, up to 55°). All three unit slips have the specified physical signs, and `dip > 90` puts the hanging wall on the left. | S2a/S2b/S2c. |
| 4 | **`dense_no_meshes` ran the meshed dense solve**, and the no-mesh path it was supposed to run crashed on any model with strain blocks or Mogi sources and weighted block-rotation constraints with 1 instead of 1e24. **Fixed.** | S8; D1, D3. |
| 5 | **Every dense run wrote its outputs twice, and only dense runs ever plotted.** **Fixed** (main writes once and plots for every solver; the plot follows the estimation's structure, not the config label). | D2. |
| 6 | **`celeri-forward` wrote `model_*_vel_tde` with the opposite sign to `model_station.csv`** (issue #508). **Fixed in celeri-forward**; the contract is documented: `model_*_elastic_segment` and `model_*_vel_tde` are the raw elastic velocities of the estimated slip, and total = rotation − elastic_segment − tde + strain + Mogi. | S4a (decomposition identity to 1e-14 for dense, eigen, qp2 and MCMC), S4b (every other column reproduced to 4 decimals). |
| 7 | **The MCMC forward model equals the linear operator** at a random point to 1.3e-5 mm/yr (float32 operands); the rotation unit chain (rad/Gyr → mrad/yr, the a-priori Euler pole prior, the RMS-velocity precision) is exact; `_state_vector_from_draw` and `mcmc_draw` are consistent; the Voronoi weights, censored bounds and transforms are correct. The one defect on the default path (the LOS log-likelihood summed to a scalar, breaking pointwise WAIC for LOS data) is **fixed**. | S5a/S5b/S5d, R1. |
| 8 | **Block closure is sound** on both demo models: every station and block interior point lies in exactly one polygon, `V − E + F = 2`, polygon areas sum to 4π exactly, labels are consistent. | S6, now `tests/test_closure_invariants.py`-style assertions in the harness. |
| 9 | **Elastic-operator caches could serve stale TDE operators** when a mesh file changed at an unchanged path (and, in streaming mode, `force_recompute` never reached them). **Fixed** by stamping every cached TDE dataset with a digest of the mesh geometry. Note: caches written before this change carry no digest and their TDE datasets are recomputed once. | R2-1, R2-2 (reproduced by editing a mesh in place: served operator unchanged, fresh operator differed by 7e-2). |
| 10 | Two further wrong-results defects in the bounded solvers: the `qp2` zero-slip regularisation mask used a concatenated layout against an interleaved vector (penalised the wrong segments whenever flags varied), and the `qp` bound update could cross its bounds after a kinematic sign change and abort the run. **Both fixed.** | R4-1 (49.6 % of flagged components mapped correctly on a mixed-flag file), R4-2 (reproduced on test_japan with a larger reduction factor). |

Baseline before any change: all three CI test groups green (`other` 151 passed / 12 skipped,
`solve` 31 passed, `optimize` 15 passed / 1 xfail). After the fix series: see section 6.

### Before/after impact of the triangle strike/dip fix (harness S1c, S1d, S3b)

Velocity step across the trace of a fully coupled mesh (kinematic slip everywhere), as a
fraction of the relative block velocity, over transects through the mesh-tied segments:

| Mesh | Transects | Before: median / p90 / max | After: median / p90 / max |
|---|---|---|---|
| Nankai (test_japan) | 16 | 0.061 / 0.085 / 0.197 | 0.007 / 0.040 / 0.108 |
| Japan trench (test_japan) | 10 | 0.074 / 0.117 / 0.142 | 0.004 / 0.006 / 0.023 |
| Cascadia (test_wna) | 10 | 0.179 / 1.000 / 1.004 | 0.007 / 0.999 / 1.006 |

The residual maxima are transects where the segment trace sits 2–4 km from the mesh top edge
(a data issue, section 4), so the block-motion step and the elastic compensation fall on
different stations; one Nankai transect (segment `eu_phil_1bbaabb`, 23 mm/yr relative motion)
keeps a 10.8 % step for the same reason. On the synthetic 60° fault at 40° N the kinematic dip
slip is now 39.96–40.03 mm/yr against the segment's 39.92, the TDE and Okada velocity fields
agree to 0.03–0.05 mm/yr beyond 3 km from the trace (4.0 mm/yr before), and both vertex
orders of the mesh load identically. The kinematic operator of the fixed code equals the
harness's independent east-north-up computation exactly (S3b: all differences 0.00 mm/yr).

## 3. Landed fixes (one commit each on `vet`)

Severity: **W** wrong results, **C** crash, **S** silent misbehaviour, **K** contract, **H** hygiene.

| ID | Sev | Commit subject | What was wrong |
|---|---|---|---|
| D3 | W | Fix dense_no_meshes index, operator columns and block-constraint weight | Strain/Mogi column ranges sized with `n_slip_rate_constraints`; block-only operator had no strain/Mogi columns; block-constraint weight hard-coded 1.0. |
| D1 | W | Honor tde/eigen in `_build_and_solve` so dense_no_meshes solves without meshes | Arguments ignored; both dense drivers ran the meshed solve. |
| D2 | S | Write outputs once and plot from celeri_solve.main for every solve type | Double `write_output`; plotting only for dense; `plot.py` keyed on the `solve_type` string. |
| D4, D5 | W | Report Mogi and Euler pole uncertainties from the estimated covariance | `volume_change_sig` held the rates; `euler_*_err` were always the zero-covariance result. Now propagated from the covariance (dense) or the posterior (MCMC), NaN otherwise. |
| D6 | K | Keep segment column names in the HDF5 output | `attrs["columns"]` overwritten; added `segment_columns`, `station_columns`, `*_index`. |
| D7 | C | Compute SAR derived columns for non-empty frames | Branches inverted; a SAR file crashed `process_sar`. SAR is consumed by no solver (warning added). |
| D8 | C | Restore `locking_depth_override_value` on Config | The override flag read a field that no longer existed. |
| D9 | S | Require mesh_file_index < number of meshes when zeroing locking depths | Off-by-one zeroed the locking depth of a segment pointing at a non-existent mesh. |
| D10 | K | Constrain solve_type to the supported solvers | Default `"hmatrix"` was not a solver; no validation. |
| D11 | K | Remove dead CLI options and validate CLI overrides | Five options silently dropped; overrides bypassed validation. |
| D12 | W | Count only segment-tied meshes in the SQP convergence criterion | Denominator counted every mesh → optimistic early exit. |
| D13 | S | Keep both slip components in the out-of-bounds trace; Pack per-element out-of-bounds counts as (strike, dip) before the totals | `qp2` trace kept strike-slip counts only, and `SlipRateLimitItem.out_of_bounds_detailed` returned (count, total) pairs, so the dip-slip entry was the element total at every iteration (found when the summed trace read count + total on the reference Japan model). The convergence decision uses `out_of_bounds()` and was never affected. |
| D14 | C | Save diagnostic block-closure figures instead of calling plt.show | Blocking GUI calls inside non-interactive solves (multi-hour hangs on macOS). |
| D15 | S | Keep Cartesian segment endpoints consistent with ordered endpoints | `x1..z2` pointed at the opposite ends of reordered segments. |
| D17 | S | Compute eigen_to_tde_bcs when operators are built | Built only as a side effect of the dense-operator getter; MCMC runs serialised an empty dict. |
| D18 | C | Fix interior-point search control flow in Polygon | Raised when the only valid point was on the last edge; silent fall-through otherwise. |
| D19 | H | Count TDE constraints arithmetically and build constraint matrices directly | Dense (2n)² scratch matrices (12.8 GB at 20k elements), twice. |
| R2-3, R3-2, R2-12 | W | Apply block strain rates as a symmetric strain in a local east-north frame | See headline 2; block centroid also fixed across the 0/360 meridian. |
| R4-1 | W | Regularize the flagged slip-rate components in the interleaved layout | See headline 10. |
| R4-2, R4-3, R4-9 | C/K | Keep SQP bounds ordered, check the presolve status and scope cvxopt options | Bound crossing; presolve status unchecked; `celeri_version` missing on qp estimations; process-wide cvxopt options mutated; misattributed xfail. |
| R4-14, R4-11 | S/C | Guard qp2 parameter scaling against all-zero columns | Zero column → NaN problem; debug plot loop indexed past two columns. |
| R2-7, R3-5, R3-8, R3-10, R2-4, R3-6, R3-11 | C/S | Validate inputs that used to fail late or silently | Mesh-free models crashed `qp`/`qp2`/`mcmc` operator builds; positive-down mesh depths and too many eigenmodes loaded silently or failed opaquely; unordered/typo'd `ScalarBound`; slip-rate constraints on components block motion cannot produce; `"none"` file-name override crashed; cross-field config rules skipped after CLI overrides. |
| R2-6, R3-9, R2-16 | S | Promote interleaved dtypes and skip smoothing of isolated mesh elements | Integer strike-slip columns truncated float dip/tensile values; isolated elements produced inf/NaN in the Laplacian. |
| D26, D25, R3-3 | W | Compute triangle strike, dip and centroids in a local east-north frame | See headline 1; also centroids across the 0/360 meridian. The commit also reordered every triangle to an upward normal; that part changed the stored dip-slip sign on every segmesh mesh and was reverted on 2026-09-15 (section 8). |
| D23 | K | Make celeri-forward's TDE columns match model_station.csv | See headline 6. |
| R1-1, R1-3, R1-5, R1-7, R1-2, R1-6 | S/K | Keep the LOS log-likelihood pointwise and validate MCMC inputs | LOS logp summed to a scalar; `random=` callables returned distributions; zero/missing `*_rate_sig` on flag-1 constraints; boundary flag 2 silently ignored by MCMC; `direct`/`low_rank` prediction mismatch now warned about; wrong prior-mean comment. |
| R4-6, R4-7 | K | Write the whole config to the HDF5 output and NaN for undefined mesh fields | Paths, booleans and ranges were dropped from the HDF5 `config` group; `model_meshes.csv` wrote zeros where the HDF5 omitted fields; coupling ratios raised warnings on zero kinematic rates. |
| R2-1, R2-2 | W | Validate cached TDE operators against the mesh geometry | See headline 9. |

Commit series on `vet` (oldest first):

- `fb9fdd5` Fix dense_no_meshes index, operator columns and block-constraint weight
- `7b44eb4` Honor tde/eigen in _build_and_solve so dense_no_meshes solves without meshes
- `13bcfb5` Write outputs once and plot from celeri_solve.main for every solve type
- `da70717` Report Mogi and Euler pole uncertainties from the estimated covariance
- `a2d005e` Keep segment column names in the HDF5 output
- `cebc67a` Compute SAR derived columns for non-empty frames
- `6464c4b` Restore locking_depth_override_value on Config
- `770aee0` Require mesh_file_index < number of meshes when zeroing locking depths
- `8bc9147` Constrain solve_type to the supported solvers
- `fb99ba1` Remove dead CLI options and validate CLI overrides
- `d921344` Count only segment-tied meshes in the SQP convergence criterion
- `2947c26` Keep both slip components in the out-of-bounds trace
- `7bf2848` Save diagnostic block-closure figures instead of calling plt.show
- `6e2e8d5` Keep Cartesian segment endpoints consistent with ordered endpoints
- `c9e4d10` Compute eigen_to_tde_bcs when operators are built
- `efc608f` Fix interior-point search control flow in Polygon
- `fce0ad6` Count TDE constraints arithmetically and build constraint matrices directly
- `5e1e315` Apply block strain rates as a symmetric strain in a local east-north frame
- `f893813` Regularize the flagged slip-rate components in the interleaved layout
- `f245fde` Keep SQP bounds ordered, check the presolve status and scope cvxopt options
- `1f57ee3` Guard qp2 parameter scaling against all-zero columns
- `14b3793` Validate inputs that used to fail late or silently
- `fe92702` Promote interleaved dtypes and skip smoothing of isolated mesh elements
- `c56f493` Compute triangle strike, dip and centroids in a local east-north frame
- `0156deb` Make celeri-forward's TDE columns match model_station.csv
- `97193f4` Keep the LOS log-likelihood pointwise and validate MCMC inputs
- `69e0e55` Write the whole config to the HDF5 output and NaN for undefined mesh fields
- `25ed9e4` Validate cached TDE operators against the mesh geometry
- `1d1453a` Regenerate the WNA eigen and TDE solution baselines for the normalised winding
- `59b7874` Add the September 2026 correctness audit report of the celeri-solve path
- `bd3af79` Pack per-element out-of-bounds counts as (strike, dip) before the totals

Baselines regenerated: `test_dense_sol_test_japan_config-{False-False,False-True,True-True}.txt`
(no-mesh columns and the strain-rate operator), `test_operator_rotation_to_tri_slip_rate_*.txt`
(triangle strike/dip), and, for WNA, `test_operator_eigen_to_velocities` plus the two solution
baselines that include TDE slip (the normalised winding of the one reversed Cascadia triangle).
Strike and dip do not enter the elastic operators, so no Okada or TDE operator baseline changed
for Japan. The winding is not purely cosmetic for a regularised solve: the Laplacian smoothing
assumes one sign convention across the mesh, so the reversed triangle had been smoothed towards
its neighbours' *numbers* while carrying the opposite physical sense (normal slip of 15 mm/yr
among reverse-slipping neighbours, coupling −0.43). Predicted station velocities change by at
most 0.009 mm/yr and the residual RMS is unchanged, because that deep triangle barely reaches
the stations. Note that the solve-group run made right after the winding commit had loaded the
stale cached operator and passed; the cache-validation commit exposed it, which is the defect it
was written to catch. **The reordering was reverted on 2026-09-15 and these WNA baselines were
regenerated again with the file's winding (section 8); a mixed-winding mesh now produces a
warning instead.**

## 4. Proposed, not landed (science or design decisions)

| ID | Where | Finding | Evidence | Proposal |
|---|---|---|---|---|
| D22 | `operators.py` eigen boundary rows | `eigenmode_slip_rate_constraint_weight` is multiplied into the operator **and** `bottom_slip_rate_weight` is applied in the weight vector, giving an effective weight w_b·w_e²; `top_slip_rate_weight` and `side_slip_rate_weight` are never used; the dense (non-eigen) path uses `tri_con_weight` instead; the docstrings say the eigen weight replaces the three. | S7 on test_wna (w_e = 10, w_b = 100): solutions with (10, 100) and (1, 1e4) identical to 1e-12; dropping the operator-side factor changes the solution by 95 % in norm (bottom-edge slip 0.02 → 0.14 mm/yr). | Decide the intended semantics (one weight, applied once, in the weight vector) and honour the per-boundary weights. |
| D24 | `operators.py:2477-2506`, `mesh.py` | Boundary constraint value 2 (tie boundary slip to the kinematic rate) exists only in the non-eigen dense operator; the eigen path treats it as zero slip; MCMC now rejects it. | Code reading. | Implement for eigen/MCMC (coupling = 1 rows) or document as dense-only. |
| R3-1 | `mesh.py:546-600` boundary classification | Lateral-edge elements with an up-dip opposite vertex are classified as top/bottom; the histogram guard mis-measures the gap and uses an absolute 10 km threshold. Sagami: two elements at 0–8 km depth flagged as bottom and constrained to zero slip. | Element 129 (vertices −8/0/−4 km) in `bottom_elements`. | Classify perimeter *edges* by their direction relative to the local dip direction (along-strike edges are top/bottom, down-dip edges are sides). |
| R4-8, R2-13 | `solve.py:242-263`, `operators.py:1221-1245` | Kinematic rates and couplings written to `model_meshes.csv` are Gaussian-smoothed (length scale hard-coded 0.25° in lon/lat degrees, anisotropic by cos(lat)) whenever eigen operators exist, unsmoothed for the plain dense solve; `qp2` couplings are bounded with respect to the smoothed rate. | Japan: smoothing changes strike-slip kinematic rates by up to 59 mm/yr and flips its sign on 5–8 % of elements; unsmoothed vs smoothed coupling differ by a median of 0.16–1.36. | Put the length scale in `MeshConfig` in km (ECEF distances) and write both kinematic columns, or name the smoothed one. |
| D35, R1-8 | `solve_mcmc.py` | The MCMC station likelihood uses one global learned sigma and ignores the per-station `*_sig` columns; block strain and Mogi carry informative priors; LOS is Gaussian while stations are Student-t. None of this is documented in `config.py`. | R1 cross-solver table (section 5). | Document; consider `sigma_i = scale · east_sig_i`. |
| R1-2, R1-4 | `solve_mcmc.py:495-531`, `:1868` | With `direct`/`low_rank` the sampler fits the full elastic field but every output uses its eigenmode projection (7.5 % of |mu| on WNA); with the default `project_to_eigen`, the likelihood sees only V·Vᵀ·e while `model_meshes.csv` reports the full posterior-mean field (out-of-span fraction 0.40 for a coupling-mode prior draw). | S5a, S5c, R1. | Store the posterior-mean `mu` on the estimation for the non-default methods; log ‖(I−VVᵀ)e‖/‖e‖ per mesh. |
| R2-5 | `spatial.py:395-401`, `operators.py:1921` | `dip == 90` is an exact float test; 89.5° amplifies fault-normal motion into dip slip by ×115. A vertical mesh gives 1/cos(90°) = 1e16. | `tests/data/segment/wna_segment0.csv` has 92°, 89.5°, 89°, 88.5° segments. | Tolerance for "vertical"; warn above a factor of ~10; vertical branch for TDEs. |
| R4-4 | `solve.py` dense solve | Every dense solve emits `LinAlgWarning` (rcond 1e-27 … 1e-36) from column scaling (Mogi ≈ 1e-20, smoothing ≈ 9e14); the solution is accurate to 1e-7 mm/yr against a scaled reference. | R4 E1. | Solve the Jacobi-scaled normal equations. |
| R4-5, D20 | `solve.py`, `solve_mcmc.py:1788` | The full `state_covariance_matrix` (677 MB of a 977 MB Japan run folder) is written by every run; `pm.compute_log_likelihood` runs unconditionally. | R4 E3. | Opt-in covariance; flag for the log-likelihood (default true to keep the WAIC filter working). |
| R3-4, R3-7 | `mesh.py:437-472`, `celeri_closure.py` | Meshes with holes or several components are silently mis-classified (only one perimeter loop is walked); dangling or duplicate segments produce unhelpful errors and a PNG in the working directory. | Synthetic meshes; 5e-7° endpoint mismatch. | Collect all loops / raise with the element; pre-flight vertex degrees with segment names. |
| D21 | `config.py:706` | Every `get_config` call creates a numbered run folder, including tests and failed runs. | Empty folders in `runs/`. | Create lazily in `build_model`. |
| Data hygiene | `data/segment/*.csv` | Closure segments thousands of km long carry `locking_depth = 15` (and 35) and therefore contribute Okada velocities from a planar rectangle spanning a hemisphere; Cascadia trace segments sit 2–4 km from the mesh top edge. | S2b, S1c. | Set closure locking depths to 0; snap mesh-tied traces to the mesh edge. |

## 5. Verified correct and cross-solver differences

Verified by the harness or the review lanes (in addition to the headline items): the
decomposition identity for all four solvers; `model_meshes.csv` equals the HDF5 mesh datasets
for all four solvers; the a-priori Euler pole constraint (JDF block) is honoured exactly and the
pole/rate round-trip is correct; the umbrella-Laplacian smoothing matrix (row sums zero,
units 1/m, no strike/dip coupling); the Mogi operator (positive = inflation); the QP inequality
packing; the eigen-layout column indices for strain and Mogi; `to_disk`/`from_disk` round trips;
the `RotationTransform` (exact inverse, Jacobian handled); the eigenmodes (orthonormal,
positive, `eigh`/`eigsh` agree); `celeri-forward`'s per-batch rebuild of every operator.

Differences between the MCMC forward model and the dense/qp forward model that are not stated in
`config.py` (from lane R1): station weights (1/σ² vs global σ with Voronoi weights); flag-1
segment observations (uniform `slip_constraint_weight` vs `Normal(σ = *_rate_sig)`); mesh
regularisation (Laplacian / eigen truncation vs truncation + Matérn prior; `smoothing_weight`
unused by MCMC); the likelihood sees only the eigen-projected elastic field; boundary
constraints as soft observations with σ 0.5 vs weighted rows; informative strain and Mogi
priors; LOS Gaussian vs station Student-t.

## 6. Tests

New or extended tests (all fail on `main` where they test a fix): `test_solve_dense.py`
(no-mesh layout, driver flags, Mogi sigma, Euler errors, eigen boundary operator, constraint
matrices, mesh-free operator build, zero-effect constraints), `test_output_files.py` (single
write, plotting for every layout, HDF5 attributes and config group), `test_model.py`,
`test_config.py`, `test_mesh.py`, `test_mesh_geometry.py`, `test_strain_rate_operator.py`,
`test_celeri_util.py`, `test_forward_contract.py`, `test_closure.py` (interior point, debug
plot), `test_optimize.py`, `test_optimize_sqp.py`, `test_process_args.py`, `test_solve_mcmc.py`,
`test_cache.py` (mesh geometry validation).

Coverage gaps that remain (from the mapping pass): `tests/test_cli.py` is never run in CI; the
`direct`/`low_rank` MCMC methods, the LOS likelihood, the MCMC censored bounds and the flag-1
rotation prior have no tests; `plot_estimation_summary` is only smoke-tested; `qp` has no
convergence-quality assertion (a run that hits `max_iter` returns without a flag, R4-10).

Final test status on `vet` (run locally with `MPLBACKEND=Agg`, the CI split): `other` 191 passed /
12 skipped, `solve` 41 passed with `--arraydiff`, `optimize` 21 passed / 1 expected failure.
Before the audit: 151 / 12, 31, 15 / 1.

## 7. Reproducing the numerical checks

The harness scripts (`s0_geometry.py`, `s2b_okada_reference.py`, `s6_closure.py`,
`s1_kinematics.py`, `s1c_transects.py`, `s1d_synthetic.py`, `s1d_fixed.py`,
`s7_eigen_weights.py`, `s5_mcmc_identity.py`, `s9_solve_types.py`, `s4b_forward.py`) and their
outputs are in the audit session's scratchpad; each is a standalone script that builds the
models in-process, uses a private elastic-operator cache, and prints PASS/FAIL with the numbers
quoted above. The durable checks were promoted to tests (section 6); the synthetic model
generator in `s1d_synthetic.py` is the template for `tests/test_mesh_geometry.py`.

## 8. Correction of 2026-09-15: the vertex reordering changed a sign convention, and was reverted

The commit `c56f493` did two things: it measured strike and dip in a local east-north-up frame
(the fix of headline 1, kept) and it reordered the vertices of every triangle whose right-hand
normal points down so that all normals point up. The second part was described as normalising
"one reversed Cascadia triangle". It did much more. The segmesh tool writes every ribbon mesh
with downward normals (a Taiwan segmesh: 126 of 126 triangles; WNA meshes 15, 45 and 63: every
triangle), so the reordering reversed every triangle of every mesh the group makes. On such a
triangle the old code reported the dip in (90, 180], the kinematic factor 1/cos(dip) was negative
and cutde's dip-slip column had the opposite sign, so the stored dip-slip number was negative for
reverse motion; the two signs cancel and the velocities were right either way. Reordering flips
both, which is physically neutral (verified on WNA mesh 15 with cutde: strike-slip and tensile
columns unchanged to 1e-14, dip-slip column exactly negated; kinematic dip slip negated) but
changes the stored dip-slip sign of every kinematic and elastic rate, of the state vector and of
the MCMC `elastic_*_ds` fields against every earlier run, and makes asymmetric
`elastic_constraints_ds` bounds refer to the opposite physical sense. Changing a sign convention
was never a goal of the audit and the effect was not measured or reported. Brendan's decision:
revert the reordering.

| Quantity | Before `c56f493` | With the reordering | After the revert |
|---|---|---|---|
| Vertex order in `mesh.verts` | As in the .msh file | Reversed on every downward-wound triangle | As in the .msh file |
| `mesh.dip` on such triangles | 90 to 180 deg | 0 to 90 deg | 90 to 180 deg |
| `mesh.strike` on such triangles | Strike + 180 deg | Strike | Strike + 180 deg (the kinematic code reduces strike mod 180) |
| Kinematic strike-slip rate | | Unchanged | |
| Kinematic dip-slip rate | Sign follows the file's winding (negative = reverse on downward-wound triangles) | Positive = reverse everywhere | Sign follows the file's winding |
| cutde strike-slip and tensile columns | | Unchanged | |
| cutde dip-slip column | | Negated on reordered triangles | As before |
| Station velocities for a given physical slip | | Identical | |
| Stored dip-slip rates, state vector, MCMC `elastic_*_ds`, meaning of asymmetric `elastic_constraints_ds` | File's convention | Opposite sign on downward-wound meshes | File's convention |
| Mixed-winding meshes (WNA mesh 48: 463 of 689 down; Cascadia: 1) | Per-triangle sign inconsistent | Consistent | Inconsistent, now reported by a warning naming the mesh and the counts |

The east-north-up strike/dip, the Cartesian-centroid longitude/latitude, the negative-depth check
and the cache geometry digest are kept. `tests/test_mesh_geometry.py` now checks that the file's
winding is preserved, that a reversed strip reports the same plane with dip in (90, 180] and the
opposite strike, that the two dip-slip sign factors reverse together (cutde columns), and that a
mixed-winding mesh warns.

### The "stripes" of September 2026 were the output change of PR #507, not the geometry

A Longitudinal Valley fault MCMC run made after PR #515 showed along-strike stripes and
per-triangle speckles in the kinematic, elastic and slipping rates that an earlier run did not.
The east-north-up strike/dip is not the cause: on nine meshes (Japan, Cascadia, NSHM, WNA ribbon
and Taiwan segmeshes) the new strike differs from the old by a near-constant per-mesh offset
(1.2 to 6.3 deg) and dip by at most 2.4 deg, and neighbour-to-neighbour roughness of strike and
dip is the same before and after. The stripes are the true per-element kinematic rates of a gmsh
ribbon mesh (adjacent triangles differ in strike by up to 20 deg and in dip by up to 10 deg, and
1/cos(dip) amplifies the dip jitter), which the MCMC sampler has always multiplied coupling by.
What changed is the output: until PR #507 (`fa386e8`, 2026-08-26) eigen and MCMC runs wrote the
Gaussian-smoothed kinematic rate (0.25 deg lon/lat kernel) and the eigen-projected elastic
field; from then on MCMC wrote the raw per-element rate and coupling x raw kinematic. On the WNA
MCMC run 84 (July 2026) the stored kinematic rates equal the Gaussian-smoothed rates of the old
code to 0.000 mm/yr on every dipping mesh, while the raw rates differ by up to 80 mm/yr in dip
slip. The remedy (this correction) makes the smoothing a per-mesh configuration
(`MeshConfig.kinematic_smoothing_length_scale`, km over the straight-line centroid distance,
default `Config.mesh_default_kinematic_smoothing_length_scale` = 25 km, 0 disables) applied to
the kinematic operator itself, so that the MCMC coupling model, the SQP bounds, the dense
boundary rows and every output share one smooth kinematic field; the raw per-element rates are
written alongside as `*_kinematic_raw`. The dip-slip rows are smoothed in the convention of an
upward-wound element (each element's winding sign is divided out before and multiplied back
after), so a mixed-winding mesh is smoothed physically rather than cancelling opposite signs;
the stored signs are untouched. The degree-based `iterative_coupling_smoothing_length_scale` is
deprecated and converted with a warning, and the new field is excluded from the elastic-operator
cache key. Two consequences are worth knowing: the smoothing spreads the 1/cos(dip)-amplified
dip-slip rate of a near-vertical element over its neighbours (the pre-existing proposal R2-5
remains the fix), and operators saved before this change carry only the unsmoothed operator
(`Operators.from_disk` warns).
