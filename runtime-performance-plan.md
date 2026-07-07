# celeri runtime-performance survey and improvement plan

*Survey date: 2026-07-06, on the state of `main` after PRs #487 (issue #485, streaming TDE
eigen operators) and #488 (streamed Okada operator) were merged. Written as a handoff so a
future Claude Code session (or human) can execute without re-surveying. No code has been
changed for any item below.*

## Measured baseline (100k WNA model: 100,592 stations, 3,978 segments, 186 meshes, 133,945 triangles; 16-core / 128 GB Apple-silicon Mac)

| Phase | Wall time |
|---|---|
| `build_model` | 77–96 s |
| Full streaming `build_operators` (warm okada cache) | 64 min |
| Selective okada recompute, 1 edited segment | 16 s (~5 s compute + ~11 s HDF5) |
| MCMC | not directly profiled; structural findings below |

## Ground truths that shape the strategy (verified in code/installed packages)

1. **cutde's C++ backend already OpenMP-parallelizes every `disp_matrix`/`disp` call across
   all cores** (grid = n_obs × n_src; `max_block_size=1`). The elastic kernels are core-saturated.
   → Multiprocessing over triangles/segments/meshes would oversubscribe and *slow down*.
   **Do not add process parallelism** unless worker thread counts are pinned — and even then
   the win is limited to the serial pyproj share, which cheaper fixes address.
2. No GPU backend installed (no pyopencl/pycuda); BLAS is Apple Accelerate (multithreaded);
   `blas 2.141 newaccelerate` per pixi.lock.
3. The dominant *serial* costs in the 64-min operator build: a fresh `pyproj.CRS.from_proj4`
   + `Proj` construction per triangle (~134k CRS parses ≈ 1–2 min) and per-triangle projection
   of ALL 100k obs points (n_tris × n_obs ≈ 1.35×10¹⁰ single-threaded tmerc transforms ≈
   13–25 min), plus per-call Python/launch overhead. Kernel FLOPs are the irreducible rest.
   Evidence: the #487 3-calls→1 fix gave ~3×, which only overhead-dominated loops can do.
4. MCMC forward path is already float32 (`_operator_mult`); default `mcmc_chains=1` so no
   BLAS-thread oversubscription today; `full_dense_operator` is a cached_property (no
   accidental rebuilds).
5. Scope decisions assumed (Brendan was AFK when asked — **confirm before executing**):
   prioritize operator construction + MCMC; tolerance-preserving changes by default, anything
   numerics-shifting needs individual sign-off; no multiprocessing/GPU work.

---

## Tier 1 — recommended implementation set (bitwise or fp-neutral; grouped into 3 PRs)

### PR A: hot-loop fixes in solve/MCMC

| # | Where | Problem | Fix | Effect | Numerics |
|---|---|---|---|---|---|
| A1 | `solve.py:189` (in the mesh loop starting ~:171) | `self.mcmc_trace.posterior.mean(["chain","draw"])` — a multi-GB reduction over the whole posterior — is evaluated once **per mesh** (186×) inside `mesh_estimate` | Hoist above the loop; optionally reduce only the `coupling_{i}_ss/ds` vars actually read | ~185× on this function; minutes → seconds at 100k | bitwise |
| A2 | `solve_mcmc.py:338` | `_operator_mult` does `operator.astype("f").copy(order="F")` — two full copies of every baked constant (~6 GB transient across the eigen dict at 100k) | `operator.astype(np.float32, order="F")` (single pass) | ~2× less graph-build copying, −3 GB peak | bitwise |
| A3 | `solve_mcmc.py:1779-1784` | `pm.compute_log_likelihood` runs unconditionally: a second graph compile + a chains×draws×2·n_sta pointwise-LL cube (≈1.6 GB per 1000 draws) even when WAIC/LOO is never used | Gate behind a config flag; keep enabled for the `filter_mcmc_chains_by_waic` path | Removes a compile + GBs of eval from every non-WAIC run | neutral (feature flag) |
| A4 | `optimize.py:846` | `np.testing.assert_allclose(q.T @ q, eye)` — a 2·m·n² ≈ 2.4e13-FLOP GEMM streaming a ~24 GB QR factor twice, purely to assert LAPACK orthogonality | Delete or hide behind a debug flag | Removes that cost from every `qr_sum_of_squares` build | neutral |
| A5 | `optimize_sqp.py:461-464` (legacy `solve_sqp`) | `weighted_operator = full_dense_operator * sqrt(w)` (~24 GB) rebuilt every SQP iteration though loop-invariant | Hoist out of the loop (modern `solve_sqp2` already avoids this via cvxpy Parameters) | −24 GB alloc × max_iter | bitwise |
| A6 | `solve.py:859-872` | `operator.T * weighting_vector` materializes a ~24 GB transposed temp; `state_covariance_matrix = linalg.inv(...)` adds a second O(n³) factorization even when uncertainties aren't wanted | Scale once (`Cw = operator * sqrt(w)[:,None]`, form `Cw.T @ Cw`); make covariance opt-in (field is already Optional-typed — check consumers) | Removes a giant temp + an O(n³) step per dense solve | point estimate bitwise |

### PR B: model building + plotting

| # | Where | Problem | Fix | Effect | Numerics |
|---|---|---|---|---|---|
| B1 | `celeri_closure.py:538-549` | The bounding-box short-circuit in `Polygon.contains_point` is **explicitly disabled** (triple `TODO: ... turned off!`); every one of ~96 polygons runs the full great-circle crossing test against all 100k stations (~8×10⁸ edge-point tests, multi-GB temporaries) | Re-enable the commented guard: `if self.area_steradians < 2π: is_in_bounds = self.bounds.contains(lon, lat)` | ~5–20× on station labeling; ~10–25 s off build_model | bitwise by construction (bbox only excludes provably-outside points; `BoundingBox.contains` has the meridian/hemisphere guards at :92-101) — but it was deliberately disabled, so validate label parity on the 100k model and ask Brendan why it was turned off |
| B2 | `mesh.py:672-731`, called from `from_params` at :1241 | Matérn covariance (O(n²)) + `eigh` (O(n³)) recomputed from scratch on **every** `build_model`; no cache. Also `pdist` computed twice per mesh (:700 and :749) | Disk-cache eigenvalues/eigenvectors keyed by hash(mesh vertices, matérn params, n_modes, algorithm) — same pattern as the elastic-operator cache; dedup the second `pdist` | Warm build_model loses its dominant term; also freezes eigsh sign nondeterminism within a cached lineage | bitwise on reload |
| B3 | `plot.py:374-669` (`plot_estimation_summary`, auto-run per solve, default True at `config.py:182`) | 6 quiver panels × 100k arrows + 11 × 3,978 scalar `plt.plot` calls in `common_plot_elements` + 133k filled triangles, saved at dpi=500 as PNG **and** vector PDF — minutes per solve | `rasterized=True` on quiver/fill artists; one `LineCollection` instead of the 44k-call segment loop; configurable dpi | Minutes → seconds per solve | diagnostic figure only |
| B4 | `model.py:334-340, 358-365, 446-479` | ~16k scalar `GEOID.inv/npts/fwd` calls + ~100k scalar `.loc[i,col]` ops in `process_segment`/`segment_centroids` | pyproj `Geod` accepts arrays — vectorize all three loops | ~10× on this section (a few seconds of build_model) | identical values |
| B5 | `scripts/celeri_forward.py:80-86` | Full block closure + 96 spherical polygons rebuilt **once per 1000-station batch** (~100× at 100k) though purely segment-dependent | Build closure once before the batch loop; per batch only `closure.assign_points` | Removes ~100 redundant closure builds from forward runs | identical |

### PR C: cache I/O + okada assembly

| # | Where | Problem | Fix | Effect | Numerics |
|---|---|---|---|---|---|
| C1 | `operators.py:1527-1532` (dataset create) + the `r+`/stream open sites | Okada cache dataset uses `chunks=True` (h5py auto → tall 2-D chunks); a 3-column selective write touches partial chunks over the full column height (read-modify-write ~10–40×) with the default 1 MB chunk cache thrashing | Explicit `chunks=(R, 3)` with R ≈ 8k rows (~196 KB chunks); open with `rdcc_nbytes` 64–256 MB for selective writes | Selective recompute 16 s → ~6–7 s; also removes amplification from the cold column-fill | bitwise (storage layout) |
| C2 | `spatial.py` okada slab helper → `get_okada_displacements` (:264-270) | The oblique-Mercator CRS build **and** the projection of all 100k obs happen 3× per segment (once per one-hot slip type) — ~12k redundant CRS builds + 3× redundant projections per full build | Hoist projection + projected obs once per segment; reuse for the 3 slip calls | Exactly 3× less okada pyproj work; with C1, selective step → ~4–5 s | bitwise (same projection object, same inputs) |
| C3 | `operators.py:2466` | `smoothing_matrix[i].toarray()[idx][:,idx]` densifies the full (3n×3n) sparse matrix (~7 GB at n=10k) just to fancy-index | Slice the CSR first, densify the small block | Avoids a giant temp in dense-operator assembly | bitwise |

## Tier 2 — follow-ups needing Brendan's individual sign-off (file as GitHub issues)

- **T2.1 Shared-projection triangle batching** — THE big lever for the 64-min TDE path.
  Group compact runs of triangles (the existing 1 GiB budget already implies ~138-triangle
  slabs at 100k stations) under ONE shared transverse-Mercator projection (tangent point =
  group centroid) and ONE `disp_matrix` call for the whole group. ~130× fewer pyproj
  transforms + CRS builds and far fewer kernel launches → **~1.6–2.3× on the TDE build
  (64 → ~28–40 min)**, more if `tde_operator_memory_gb` is raised (see T2.6). **Numerically
  different**: a triangle offset d from the shared tangent point is rescaled by ~(d/R)²/2 →
  ~1e-5–1e-4 relative in source geometry for ≤50–100 km groups. Keeps pyproj's exact
  ellipsoidal tmerc — only the tangent point moves. Needs: proximity grouping (mesh index
  order is usually already compact; Morton-sort as fallback), `(n_obs,3,B,3)` reshape into the
  slab layout, and a validation study vs the current operator (station velocities, eigen
  projections). Effort ~1–2 days + validation. **This is a science decision.**
- **T2.2 9×-tiled okada kernel** (deferred from #488): one matched-pair `cutde.disp` call over
  3 sub-triangles × 3 one-hot slips (`okada/cutde_okada.py:349-370`), reassembling the 3-column
  Jacobian; the per-obs sub-triangle summation regrouping makes this fp-tolerance, not bitwise.
  ~1.5–2× okada compute. Pairs with C2.
- **T2.3 MCMC trace slimming**: `pm.Deterministic("mu", ...)` (`solve_mcmc.py:1641`,
  draws×201k ≈ 1.6 GB/1000 draws), `los_predicted` (:1389), and per-mesh TDE deterministics
  are stored every draw but largely unread (Estimation reconstructs velocities from
  `state_vector`). Drop the unread ones from trace storage. Identical numerics; user-visible
  trace contents change — needs a survey of downstream notebook usage.
- **T2.4 Batched eigen matmul in the MCMC graph**: ~372 skinny float32 gemv nodes
  (201k×10 each, `solve_mcmc.py:765`, summed at :1629) → one (201k×Σmodes) gemv for all
  unconstrained meshes (bounded/coupling meshes stay separate: `_constrain_field` is
  nonlinear). Big compile-time and per-draw dispatch win, especially for the jax backend;
  fp-tolerance.
- **T2.5 Far-field culling or cutde ACA** (issue #69 direction): after T2.1 the remaining
  cost is ~1.35×10¹⁰ TDE evaluations, mostly near-zero far-field. Distance cutoff or cutde's
  shipped ACA (`cutde/aca.py`) → 5–50× theoretical. Controlled approximation; needs a
  convergence study (the eigen projection may absorb much of the error — verify).
- **T2.6 Raise `tde_operator_memory_gb` default** (1.0 → e.g. 4.0, `config.py:94`): zero
  numerics, larger slabs/batches everywhere, directly amplifies T2.1. Trivial but changes a
  default on a 128 GB-class assumption.
- **T2.7 Multi-chain thread hygiene**: if `mcmc_chains>1` with numba backend, chains ×
  Accelerate threads can oversubscribe; cap BLAS threads per chain (threadpoolctl /
  VECLIB_MAXIMUM_THREADS) around `nutpie.sample`. Low priority while default chains=1.
- **T2.8 WAIC filter diagnostics**: `filter_mcmc_chains_by_waic.py:300-309` runs pointwise
  PSIS-LOO per chain only to fill log columns (the keep/exclude decision uses WAIC only);
  make LOO optional, compute WAIC intermediates float32/chunked.

## Explicitly rejected (with reasons — don't re-litigate without new facts)

- **Multiprocessing over sources/meshes**: cutde cpp kernels + LAPACK already saturate all
  cores; workers would oversubscribe (and macOS spawn re-pickles inputs per worker).
- **GPU**: no pyopencl/pycuda installed; cutde GPU float64 exists but is an infra project,
  not a code change. Revisit only if HPC/GPU targets appear.
- **eigsh/float32 eigenmodes as defaults**: basis sign/rotation changes; B2's cache gets most
  of the win bitwise-safely.
- **lru_cache on `get_keep_index_12`**: returns mutable arrays; unsafe without copy discipline,
  and it's not hot.

## Verification recipe (per PR)

1. Full suites (local, `MPLBACKEND=Agg`; ignore the 36 pre-existing `tests/okada/` failures on
   this Mac): arraydiff (`pytest tests/test_solve_dense.py --arraydiff` — must stay green with
   unchanged baselines for all bitwise-claimed items), other/optimize/legacy-mcmc groups.
2. New targeted tests: B1 label parity (old vs new `assign_block_labels` station labels equal
   on test configs + one 100k parity driver); B2 cache-hit == cold-compute bitwise; A3 flag
   default keeps WAIC-filter tests green.
3. Timing evidence in each PR: `build_model` at 100k before/after (B1/B2/B4); selective-
   recompute step via the #488 Step-D driver pattern on an APFS clone of
   `~/Desktop/wna/data/operators_bench/a7f5d8917d7ec05f.hdf5` (C1/C2); `mesh_estimate`
   wall time on an existing trace (A1).
4. Adversarial multi-agent review of each diff before finalizing (the #487/#488 reviews each
   caught real cache-lifecycle bugs — treat as mandatory for anything touching
   `_store_elastic_operators`).

## Suggested execution order

PR A → PR B → PR C, then file T2.1–T2.8 as issues (T2.1 first — it's the headline win and
needs the science conversation). Confirm the "Assumed scope decisions" with Brendan before
starting, especially: the B1 bbox re-enable (why was it disabled?), the A3/A6 behavior flags,
and whether T2.1's ~1e-4-level numerical shift is acceptable in principle.
