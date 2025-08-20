# Quasi-static imaging of earthquake cycle kinematics

`celeri` is a Python-based package designed to image earthquake cycle activity, including spatial slip deficit/fault coupling across geometrically complex fault systems at large scales. It features:

- GUI-based model building with [`celeri_ui`](https://brendanjmeade.github.io/celeri_ui/)
- Graphical comparisons of model results with [`result_manager`](https://github.com/brendanjmeade/result_manager)
- 3D visualization of model results with [`parsli`](https://github.com/brendanjmeade/parsli)
- Fast and automated block closure on the sphere
- Large aperture models with locally optimized sphere flattening
- Implicity smoothing and small memory footprint via distance-weighted eigenmodes
- Slip rate and coupling bounded solves via [sequential quadratic programming](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EA004229)
- MCMC uncertainty estimates
- Blazingly fast elastic calculations via [Ben Thompson's](https://github.com/tbenthompson) [cutde](https://github.com/tbenthompson/cutde)
- Easy IO with standard file types (`.csv`, `.json`, `.hdf5`, `.pkl`)

## Getting started

To set up a development conda environment, install [pixi](https://pixi.sh/) and run the following command in the `celeri` folder.

```bash
pixi shell
```

Alternatively, run the following commands in the `celeri` folder.

```bash
conda config --prepend channels conda-forge
conda env create
conda activate celeri
pip install --no-use-pep517 -e .
```

From here, you can launch model runs with `celeri_solve`.

To run notebooks from VSCode:

1. cd to the `celeri` folder.
2. Use  `code .` to start VSCode from the command line.
3. Navigate to the notebook you'd like to run.
4. Click on the Python environment selector near the upper right-hand corner of the VSCode window.
5. Select the "default" shell.
6. Run the notebook.

## Command line workflow

### `celeri-solve`

- Estimate model parameters.
- A `*_config.json` file is a required argument.
- With the Python environment activated, run:

```bash
celeri-solve <my_config.json>
```

- This will create a folder in in the `runs/` directory that contains all output files.  New folders are created automatically for each run and are sequentially numbered.

### `celeri-forward`

- Predict surface velocities from model parameters constrained by previous `celeri-solve` run.
- `celeri-forward` is batched so that it never creates large matrices.
- With the Python environment activated, run:

```bash
celeri-forward <path to output folder> <station file for forward model predictions>
```

- If you want to run `celeri-forward`, you probably want some gridded locations for model evaluation. That's what `create-grid-station` is for: Call as:

```bash
create-grid-station <lon_min> <lat_min> <lon_max> <lat_max> --n_points=<number of grid points>
```

- where:
  - `lon_min`: Minimum longitude
  - `lat_min`: Minimum latitude
  - `lon_max`: Maximum longitude
  - `lat_max`: Maximum latitude
  - `--n_points=<number of grid points>`: Optional. The default value is 100.
- This produces a station file (named `<UUID>_station.csv`) that can be passed to `celeri-forward`.

## Folder structure and file locations for applications

A large number of input files can be involved in a model run.  We assume that a project is arranged using the following folder structure:

```text
project_name/
├── README.md
├── notebooks/
│   ├── block_model.ipynb
│   ├── visualize_results.ipynb
│   └── resolution_tests.ipynb
├── data/
|   ├── config/
│   |   └── *config.json
│   ├── segment/
│   │   └── *segment.csv
│   ├── block/
│   │   └── *block.csv
│   ├── station/
│   │   └── *station.csv
│   ├── mesh/
│   |   ├── *mesh.json
│   |   └── *.msh
|   └── operators/
│       └── *.hdf5
└── runs/
    └── 0000000001/
       ├── 0000000001.log
       ├── output.pkl
       ├── elastic_operators.hdf5
       ├── model_segment.csv
       ├── model_block.csv
       └── model_station.csv
```

## The flow of information through celeri

![alt text](https://github.com/user-attachments/assets/d9762dce-eb82-4236-87be-d2b76e2516a4)

## Maintenance notes

See [maintenance-notes.md](maintenance-notes.md) for current best practices for maintaining this repository, in particular:

- [Cutting a new release](maintenance-notes.md#cutting-a-new-release)
- [Updating dependencies](maintenance-notes.md#updating-dependencies)

## Other earthquake cycle kinematics software

We think celeri is pretty great, but there are other great kinematic modeling:

- Jack Loveless' and Brendan Meade's MATLAB-based [Blocks](https://github.com/jploveless/Blocks)
- Rob McCaffrey's Fortran-based [TDEFNODE](https://robmccaffrey.github.io/TDEFNODE/TDEFNODE.html)
- Richard Styron's Julia-based [Oiler](https://github.com/cossatot/Oiler)
