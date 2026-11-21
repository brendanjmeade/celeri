# Correlated station noise: the `"covariance"` station weighting

## The problem

The MCMC station likelihood normally treats every station velocity as an
independent observation. That is reasonable for a GNSS network, but not for a
densely sampled velocity field, such as an InSAR-derived grid subsampled to
10⁵–10⁶ points. The residuals of such a field are correlated over kilometres,
so an independent likelihood counts each correlated patch many times over.
The weight of the data then depends on how densely the grid was sampled,
which is an arbitrary choice.

For example, in the Western North America (WNA) model, going from 100k to 1M
grid points shrank posterior standard deviations by about √10, and moved most
mesh slip rates outside the 100k run's 95% intervals. The data did not get
better; there were just ten times as many copies of it.

`"mcmc_station_weighting": "covariance"` replaces the independent likelihood
with one that accounts for this. It whitens the residuals with a sparse,
near-field correlation model:

- **Nearby stations on the same block share information.** Adding more
  stations to a patch stops adding independent data.
- **Stations farther apart than a taper radius *R*, or on different blocks,
  stay independent.** The long-wavelength velocity field stays anchored, and
  velocity steps across faults (from creep, for example) are still fit
  station by station.

Use it for densely sampled velocity fields. A sparse GNSS network does not
need it: stations tens of kilometres apart are already independent at the
default *R*.

## Quick start

1. **Run once with the unweighted likelihood.** This is the default
   (`"mcmc_station_weighting": null`, or omit the field).

2. **Fit the correlation** to that run's residuals:

   ```bash
   celeri-fit-station-covariance runs/0000000005 data/covariance/my_covariance.json \
       --station-names xu_2021
   ```

   - `--station-names` selects the stations to correlate (here a gridded
     field named `xu_2021`; GNSS stations stay independent). The default is
     every station.
   - Use the same `--taper-radius` (default 3 km) and `--same-block` /
     `--no-same-block` (default on) as the MCMC run will. The fit maximizes
     the likelihood of the tapered model, so its parameters are only valid
     for those settings. celeri warns if the run uses different ones.
   - The script writes the JSON file and a diagnostic plot next to it
     (`my_covariance.png`). See [Reading the fit output](#reading-the-fit-output).

3. **Run again with the covariance.** In the config file:

   ```json
   "mcmc_station_weighting": "covariance",
   "mcmc_station_covariance_file_name": "../covariance/my_covariance.json",
   "mcmc_station_covariance_taper_radius": 3.0,
   "mcmc_station_covariance_same_block": true
   ```

   Or with `celeri-solve` flags:

   ```bash
   celeri-solve --config_file my_config.json \
       --mcmc-station-weighting covariance \
       --mcmc-station-covariance-file-name data/covariance/my_covariance.json
   ```

4. **Check the run.** Its log reports:
   - how many stations are correlated;
   - the average number of conditioning neighbours;
   - how many independent stations the correlated set is worth for a common
     offset.

   The run folder keeps a copy of the covariance file (`station_covariance.json`)
   and the factor used (`station_covariance_factor.npz`).

## Settings

### Config fields

| Config field | `celeri-solve` flag | Default | Meaning |
|---|---|---|---|
| `mcmc_station_weighting` | `--mcmc-station-weighting` | `null` (unweighted) | Set to `"covariance"` to use this likelihood. |
| `mcmc_station_covariance_file_name` | `--mcmc-station-covariance-file-name` | none; required in this mode | The JSON file from `celeri-fit-station-covariance`. A relative path in a config file is resolved from the config file's folder. |
| `mcmc_station_covariance_taper_radius` | `--mcmc-station-covariance-taper-radius` | `3.0` (km) | Distance beyond which residuals are independent. See [Choosing the taper radius](#choosing-the-taper-radius). |
| `mcmc_station_covariance_same_block` | `--mcmc-station-covariance-same-block 0\|1` | `true` | Correlate only stations on the same block. See [Same block only](#same-block-only). |

The three `mcmc_station_covariance_*` fields are ignored unless
`mcmc_station_weighting` is `"covariance"`. The taper radius must be positive.

### `celeri-fit-station-covariance` options

| Option | Default | Meaning |
|---|---|---|
| `run_dir` | | A finished run folder; its `model_station.csv` provides the residuals. |
| `output` | | The covariance JSON file to write. The script refuses to replace an existing file unless `--overwrite` is given. |
| `--station-names NAME ...` | every station | The stations to correlate, matched on the station table's `name` column. |
| `--taper-radius` | 3 | *R* in km. Match the MCMC run. |
| `--same-block` / `--no-same-block` | on | Match the MCMC run. |
| `--n-neighbors` | 30 | Maximum number of earlier neighbours each station is conditioned on. |
| `--n-terms` | 2 | Matérn terms per component. |
| `--smoothness` | 0.5 | Matérn smoothness (0.5 is exponential). |
| `--method` | `mle` | `mle`: a least-squares fit to the correlogram, refined by maximum likelihood. `correlogram`: the least-squares fit only. |
| `--max-distance-km` | 100 | Largest distance used in the least-squares fit. |
| `--anchors` | 5000 | Number of randomly chosen stations whose pairs with their neighbours form the correlogram. |
| `--seed` | 0 | Station ordering seed, recorded in the file. |
| `--skip-convergence` | | Skip the check against more neighbours. |
| `--no-plot` | | Don't write the plot. |

## How it works

### The correlation model

For each velocity component, the correlation between the residuals of two
stations a distance *h* (km) apart is

```
rho(h) = nugget · [h = 0]
         + wendland(h; R) · Σ_k weight_k · matern(h; length_k, smoothness_k)
```

- `nugget + Σ weight_k = 1`. The nugget is noise that is uncorrelated even
  between neighbouring stations.
- The Matérn terms describe how the correlation decays with distance.
  Smoothness 0.5 gives `exp(−h/length)`.
- `wendland(h; R) = (1 − h/R)⁴ (4h/R + 1)` for *h* < *R*, and 0 beyond. This
  taper is what makes the covariance near-field. It is compactly supported
  and positive definite in three dimensions, so the tapered correlation is
  still a valid correlation.
- *h* is the straight-line distance between the stations' Cartesian
  coordinates (the station table's `x`, `y`, `z`).

The nugget and Matérn terms are fitted to an earlier run's residuals. The
station-to-station covariance itself is never stored: it is rebuilt from the
station coordinates at run time, so one file serves any subsample of the same
data set (for example, both the 100k and 1M WNA grids).

### The likelihood

The likelihood keeps the unweighted likelihood's Student-t (ν = 6) and its
single learned noise scale `sigma`, but applies them to whitened residuals.
The whitening uses Vecchia's nearest-neighbour approximation:

1. Stations are put in a fixed random order (`ordering_seed`).
2. Each station's residual is predicted (kriged) from the residuals of up to
   `n_neighbors` of the nearest *earlier* stations that are within *R* and,
   with same-block on, on the same block.
3. Only the standardized prediction error enters the likelihood:

```
e_i = (r_i − Σ_j b_ij r_j) / sqrt(d_i)
log p = Σ_i [ log StudentT(e_i; ν = 6, sigma) − ½ log d_i ]
```

Here `r` is the residual, and `b_i` and `d_i` are the kriging weights and the
kriging variance as a fraction of the total variance. A station whose
neighbours already predict its residual well has a small `d_i`. Its whitened
residual is then mostly what the neighbours did not predict, so it adds
little new information.

- **Stations without neighbours are independent.** Stations that the
  covariance file does not select (for example GNSS stations, when only a
  grid is correlated), or that have no earlier neighbour within *R*, keep
  `e_i = r_i`.
- **Cost.** The factor is built once per run: in seconds for 10⁵ stations,
  in minutes for 10⁶. Each likelihood evaluation adds one gather of the
  neighbours' residuals.
- **Rotation reparameterization.** The block-rotation SVD reparameterization
  is built from the whitened station operator, just as the `"station_sigma"`
  mode builds it from the sigma-scaled one.

### Why a taper

Without a taper, the WNA residual correlation includes a term about 13 km
long that carries 85–90% of the residual variance. Treating all of it as
noise made the entire 100k grid worth about 230 independent stations for a
common offset. A regional velocity offset of about 5 mm/yr then cost almost
nothing. An MCMC run sampled such an offset and lost the known creep on the
San Andreas creeping section and on the Calaveras and Concord–Green Valley
faults.

The taper prevents this. Correlation is limited to *R*, so every patch of
radius about *R* still anchors the long-wavelength field. At *R* = 3 km, the
WNA 100k grid is still worth about 37,000 independent stations for a common
offset.

The price: residual correlation near and beyond *R*, such as that
long-wavelength misfit, is treated as independent noise.

### Same block only

With `mcmc_station_covariance_same_block` on (the default), stations on
different blocks are independent, even when closer than *R*.

In the WNA Xu et al. (2021) residuals:

| Pairs | Distance | East correlation | North correlation |
|---|---|---|---|
| same block | 0.5–1 km | 0.73 | 0.84 |
| different blocks | 0.5–5 km | 0.4–0.6 | −0.14 to 0.05 |

- **North:** residuals are independent across faults.
- **East:** residuals remain partly correlated across faults.

Same-block-only is nevertheless the default, because the alternative breaks
the property this mode exists for. With correlation across faults, the
residual *difference* across a fault looks very precisely known, since
correlated noise cancels in a difference. That precision then grows with
sampling density. At *R* = 3 km, allowing cross-fault correlation made the
information about shallow creep on the Hayward fault 10× larger at 1M
stations than at 100k. With same-block-only correlation, the ratio was 1.9.
The cost is that cross-fault east correlation is treated as independent.

## Reading the fit output

The script prints, for each velocity component, the least-squares and the
maximum-likelihood fits, then checks the result. What each line means, and
what to look for:

- **`correlogram fit`**: a least-squares fit to the binned residual
  correlation. It is the starting point of the likelihood fit (and the
  result, with `--method correlogram`).
- **`maximum likelihood`**: the parameters written to the file, which
  maximize the Vecchia likelihood of the tapered model. The likelihood gain
  over the correlogram fit is in brackets and should be positive.
- **`length is at the fit's upper bound`**: within *R*, a correlation length
  much longer than *R* is indistinguishable from a constant, so the
  likelihood cannot pin it and the fit stops at 1000 km. The term's
  correlation then simply follows the taper's shape. This is expected when
  the residuals are correlated well beyond *R*, as in WNA. Similarly, a length
  at the lower bound (0.01 km) acts as extra nugget.
- **`whitened residual correlation within R`**: the residual correlation
  after whitening, one `[east, north]` pair per correlogram bin inside *R*.
  Values near 0 at short distances mean the near-field correlation is
  captured. Some correlation remains towards *R*, because the taper forces
  the modelled correlation to zero there while the residuals are still
  correlated: that is the correlation the model deliberately ignores (see
  [Why a taper](#why-a-taper)).
- **`independent stations for a common offset`**: how strongly the selected
  stations together still anchor the long-wavelength field. A few hundred
  for 10⁵ stations means the regional velocity is barely constrained; see
  [Why a taper](#why-a-taper).
- **`Vecchia Gaussian log density with m neighbours`**: should stop changing
  by `--n-neighbors`. If it still changes between `n_neighbors` and twice
  that, increase `--n-neighbors`.

The plot shows, for each component, the residual correlogram (dots), the
fitted correlation without and with the taper (dashed and solid lines) and
the whitened residuals' correlogram (crosses), with *R* marked.

## The covariance file

The format, with illustrative values:

```json
{
    "station_names": ["xu_2021"],
    "n_neighbors": 30,
    "ordering_seed": 0,
    "components": {
        "east": {
            "nugget": 0.05,
            "terms": [
                {"weight": 0.15, "length_km": 0.4, "smoothness": 0.5},
                {"weight": 0.80, "length_km": 18.0, "smoothness": 0.5}
            ]
        },
        "north": {"nugget": 0.05, "terms": ["...as for east..."]}
    },
    "fit": {"taper_radius_km": 3.0, "same_block": true, "...": "diagnostics"},
    "source": {"run_dir": "...", "created": "..."}
}
```

- **`station_names`**: the correlated stations, matched by name after
  stripping whitespace, or `null` for every station.
- **`n_neighbors`**: each station is conditioned on at most this many earlier
  neighbours.
- **`ordering_seed`**: fixes the random station ordering.
- **`components`**: `"east"` and `"north"`, plus `"up"` for runs with
  `include_vertical_velocity`. A missing component stays independent, with a
  warning. Each has a `nugget` and a list of `terms` (`weight`, `length_km`,
  `smoothness`). The nugget and weights must sum to 1, and the nugget must be
  positive.
- **`fit` and `source`**: informational: diagnostics, the correlograms, the
  run the residuals came from. The exceptions are `fit.taper_radius_km` and
  `fit.same_block`, which the solver compares with the run's settings,
  warning if they differ.

Hand-written files work too. Only `station_names`, `n_neighbors`,
`ordering_seed` and `components` are required.

## Choosing the taper radius

The tables below are from the WNA model. They compare the 100k and 1M
subsamples of the same Xu et al. (2021) grid, using the correlation fitted to
an unweighted 100k run's residuals, with same-block-only correlation.

**Regional anchoring.** How many independent stations the grid is worth for a
common velocity offset (mean of east and north):

| *R* (km) | 100k | 1M | 1M / 100k |
|---|---|---|---|
| independent | 99,531 | 999,951 | 10.0 |
| 1 | 84,000 | 331,000 | 3.9 |
| 2 | 57,000 | 105,000 | 1.8 |
| **3** | **37,000** | **51,000** | **1.4** |
| 5 | 18,000 | 21,000 | 1.2 |
| no taper | 230 | 280 | 1.2 |

**Fault slip.** The Fisher information about unit strike-slip patterns,
computed from the run's own elastic operators. The patterns are:

- shallow (< 3 km) slip;
- shallow slip in 5 km and 20 km windows along strike, the short-wavelength
  variation of creep;
- deep (> 10 km) slip.

They are on the Hayward, Calaveras and Concord–Green Valley faults and the
San Andreas creeping section. Each cell gives the range over faults.

| *R* (km) | Shallow: 100k vs independent | Shallow: 1M / 100k | Along-strike windows: 100k vs independent | Windows: 1M / 100k | Deep: 100k vs independent | Deep: 1M / 100k |
|---|---|---|---|---|---|---|
| independent | 1.00 | 9.7–10.6 | 1.00 | 10.1–10.8 | 1.00 | 10.0–10.3 |
| 1 | 0.83–0.85 | 4.1–4.3 | 0.83–0.86 | 4.1–4.5 | 0.84 | 3.8–3.9 |
| 2 | 0.58–0.61 | 2.0–2.3 | 0.57–0.63 | 2.0–2.5 | 0.56–0.58 | 1.8–1.9 |
| **3** | **0.41–0.44** | **1.5–1.9** | **0.42–0.48** | **1.5–2.1** | **0.36–0.39** | **1.3–1.5** |
| 5 | 0.23–0.29 | 1.3–1.8 | 0.24–0.34 | 1.3–2.1 | 0.17–0.21 | 1.1–1.2 |
| no taper | 0.89–1.67 | 3.8–7.3 | 1.22–1.76 | 4.3–7.2 | 0.01–0.29 | 3.5–6.8 |

How to read the tables:

- **A 1M/100k ratio near 1** means the result no longer depends on sampling
  density. With independent stations it is 10.
- **A taper much shorter than the correlation length does little.** At
  *R* = 1 km, 1M stations still carry four times the information of 100k.
- **Around 3 km, the two subsamples come close** (ratios 1.3–2.1, against
  10). The grid is still worth tens of thousands of independent stations for
  the long-wavelength field.
- **Short-wavelength slip keeps its share of the information.** Shallow slip
  and its along-strike variation keep somewhat more of their information
  (0.41–0.48 of the independent value) than deep slip (0.36–0.39), so known
  creep and its variation along strike remain resolvable.
- **Without a taper, the long-wavelength information collapses.** Deep slip
  keeps only 0.01–0.29 of its independent value, which is the failure
  described in [Why a taper](#why-a-taper).

*R* = 3 km is the default because it is the smallest of these radii at which
the 100k and 1M results come close. Larger radii discount the data further
for little extra agreement.

For another data set, choose *R* the same way:

- the whitened residuals' correlation within *R* should be small (see the fit
  plot);
- results should no longer change much when the data are subsampled more or
  less densely.

## Caveats

- **The correlation is estimated from one model's residuals.** It treats that
  model's systematic misfit as noise. The taper limits the effect to the near
  field. Refit after large changes to the model or the data.
- **Correlation near and beyond *R* is ignored** (see
  [Why a taper](#why-a-taper)). Posterior uncertainties are therefore still
  somewhat too small for a densely sampled field, though no longer by a
  factor that depends on the sampling.
- **LOS observations stay unweighted**, as in the `"station_sigma"` mode.
- **East–north correlation is not modelled.** The components are whitened
  independently.
- **Pointwise log-likelihoods are conditional.** Each station's
  log-likelihood is that of its whitened residual given its earlier
  neighbours, plus the `−½ log d_i` term. The values sum to the total, so WAIC
  and chain filtering work, but leave-one-station-out (LOO) estimates are
  approximate.
- **Predictive draws ignore the correlation.** Prior and posterior predictive
  draws of `station_velocity` use independent Student-t noise, like the
  `"voronoi"` mode.
- **Prefer the JAX backend for large station sets.** On JAX, the whitening
  costs about as much per gradient as the independent likelihood at 10⁵
  stations. In a standalone benchmark of the likelihood alone, numba was much
  slower: 46 ms against 0.7 ms at 10⁵ stations with 30 neighbours.
