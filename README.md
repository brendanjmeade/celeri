# Quasi-static imaging of earthquake cycle kinematics

[![GitHub stars](https://img.shields.io/github/stars/brendanjmeade/celeri?style=social)](https://github.com/brendanjmeade/celeri)
[![Tests](https://github.com/brendanjmeade/celeri/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/brendanjmeade/celeri/actions/workflows/test.yml)
[![Release pipeline](https://github.com/brendanjmeade/celeri/actions/workflows/release.yaml/badge.svg?branch=main)](https://github.com/brendanjmeade/celeri/actions/workflows/release.yaml)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/celeri.svg)](https://prefix.dev/channels/conda-forge/packages/celeri)
[![PyPI version](https://img.shields.io/pypi/v/celeri.svg)](https://pypi.org/project/celeri/)
[![pixi](https://img.shields.io/badge/pixi-project-4E54E9?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOTYiIGhlaWdodD0iOTYiIHZpZXdCb3g9IjAgMCA5NiA5NiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iOTYiIGhlaWdodD0iOTYiIHJ4PSIxNiIgZmlsbD0iI0ZGRiIvPjxwYXRoIGQ9Ik00OS45OTkgMTkuMzMzTDUwIDc2LjY2N0M2NS4yNCA3Mi43MzIgNzYuNjY3IDU5LjQ3MSA3Ni42NjcgNDMuMzMzQzc2LjY2NyAyNy4xOTUgNjUuMjQgMTMuOTM0IDUwIDEwTDUwIDYuNjY3MjNFQzMwLjIxNCAxMS4xMjIgMTYuNjY3IDI3LjE5NCAxNi42NjcgNDMuMzMzQzE2LjY2NyA1OS40NzEgMzAuMjE0IDcyLjczMiA1MCA3Ni42NjdMNDkuOTk5IDE5LjMzM1oiIGZpbGw9IiM0RTU0RTkiLz48L3N2Zz4=)](https://pixi.sh)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`celeri` is a Python-based package designed to image earthquake cycle activity, including spatial slip deficit/fault coupling across geometrically complex fault systems at large scales. It features:

- GUI-based model building with [`celeri_ui`](https://brendanjmeade.github.io/celeri_ui/)
- Graphical comparisons of model results with [`result_manager`](https://github.com/brendanjmeade/result_manager)
- 3D visualization of model results with [`parsli`](https://github.com/brendanjmeade/parsli)
- Fast and automated block closure on the sphere
- Large aperture models with locally optimized sphere flattening
- Implicit smoothing and a small memory footprint via distance-weighted eigenmodes
- Slip rate and coupling bounded solves via [sequential quadratic programming](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EA004229)
- Markov chain Monte Carlo (MCMC) uncertainty estimates
- Blazingly fast elastic calculations via [Ben Thompson's](https://github.com/tbenthompson) [cutde](https://github.com/tbenthompson/cutde)
- Easy I/O with standard file types (`.csv`, `.json`, `.hdf5`, `.pkl`)

See the [wiki](https://github.com/brendanjmeade/celeri/wiki) for more detailed information.

## Getting started

`celeri` can be run from the command line or from a Jupyter notebook.

A [project folder structure](#folder-structure-and-file-locations-for-applications) containing a `data/` directory, such as the one provided with this repository, is required to use `celeri`.

After [installation](#installation), you can use `celeri-solve` or [other commands](#command-line-workflow) from the command line:

```bash
cd data/config
celeri-solve some_config.json
```

You can also run `celeri` from a Jupyter notebook, such as those provided in the `notebooks/` directory.

To run notebooks from Visual Studio Code:

1. Ensure that `celeri` is [installed](#installation).
2. Start VS Code, and ensure that the Jupyter extension is installed.
3. Open the project folder (e.g. a clone of this repository).
4. Open the notebook you'd like to run.
5. Click on the Python environment selector near the upper right-hand corner of the VS Code window.
6. If using pixi, select the "default" kernel. Otherwise, select the environment in which you installed `celeri`.
7. Run the notebook.

If VS Code does not show an option for the "default" kernel, ensure that your Jupyter extension is up to date. You may need to restart VS Code after running `pixi shell` for the first time.

## Installation

The simplest way to install `celeri` is to use [pixi](https://pixi.sh/).

```bash
# Create a clone of this repository into the celeri/ directory
git clone https://github.com/brendanjmeade/celeri.git
cd celeri
pixi shell  # Installs and activates the pre-configured celeri "default" environment
```

Note that `pixi` is project-centric, so `pixi` commands only apply if you're within the project directory.

Pixi's support for global project installs is currently very preliminary.
In the meantime, running the following command will create shims for all your favorite celeri commands like `celeri-solve` so you can run them from anywhere:

```bash
pixi run pixi-global-install-shim-hack
```

You can remove these shims by adding the `--remove` flag to this command.

For details about how to use `pixi` in other configurations, or how to install `celeri` with other tools such as `conda`, `pip`, or `uv`, see [alternative-installation.md](alternative-installation.md).

## Command line workflow

### `celeri-solve`

- Estimate model parameters.
- A `*_config.json` file is a required argument.
- With the Python environment activated, run:

```bash
celeri-solve <my_config.json>
```

- This will create a folder in the `runs/` directory that contains all output files.  New folders are created automatically for each run and are sequentially numbered.

- All relative paths in `*_config.json` are resolved relative to the directory containing the config file. Absolute paths are used as-is.

### `celeri-forward`

- Predict surface velocities from model parameters constrained by previous `celeri-solve` run.
- `celeri-forward` is batched so that it never creates large matrices.
- With the Python environment activated, run:

```bash
celeri-forward <path to output folder> <station file for forward model predictions>
```

- If you plan to run `celeri-forward`, you may want gridded locations for model evaluation. Use `create-grid-station`:

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

```mermaid
%%{init: {
  'theme': 'forest',
  'themeVariables': { 'primaryColor': '#90EE90', 'lineColor': '#333' },
  'flowchart': { 'padding': 5, 'nodeSpacing': 15, 'rankSpacing': 55 }
}}%%

flowchart TB
    command["config.json"]
    celeri_ui["celeri_ui"]:::tool
    
    station["station.csv"]
    los["los.csv"]
    mogi["mogi.csv"]
    block["block.csv"]
    segmesh["segmesh.py"]:::tool
    segment["segment.csv"]
    mesh_json["mesh_params.json"]
    mesh_1["mesh_1.msh"]
    mesh_2["mesh_2.msh"]
    mesh_3["mesh_3.msh"]

    mesh_n["mesh_n.msh"]
            
    %% Solid lines from command.json
    command --> station
    command --> los
    command --> mogi
    command --> block
    command --> segment
    command --> mesh_json
    
    %% Solid lines from celeri_ui
    celeri_ui -.-> block
    celeri_ui -.-> segment
            
    %% Solid line segment to mesh.json
    segment --> mesh_json
    
    %% Solid lines from mesh.json to mesh files
    mesh_json --> mesh_1
    mesh_json -.-> mesh_2
    mesh_json -.-> mesh_3
    mesh_json -.-> mesh_n

    mesh_1 ~~~ mesh_2
    mesh_2 ~~~ mesh_3
    mesh_3 ~~~ mesh_n
        
    %% DASHED: segmesh to mesh files
    segmesh -.-> mesh_json
    segmesh -.-> mesh_1
    segmesh -.-> mesh_n
    segmesh -.-> mesh_2
    segmesh -.-> mesh_3
    segmesh -.-> segment
        
    celeri_solve["celeri-solve.py"]:::tool
    
    %% command.json to celeri-solve.py
    command --> celeri_solve
    
    station_out["model_station.csv"]
    segment_out["model_segment.csv"]
    output_zarr["*.zarr"]
    model_hdf["model.hdf"]
    meshes_csv["model_meshes.csv"]

    celeri_solve --> station_out
    celeri_solve --> segment_out
    celeri_solve --> meshes_csv
    celeri_solve --> model_hdf
    celeri_solve --> output_zarr
    
    fennil["fennil"]:::tool
    parsli["parsli"]:::tool
    vis_ipynb["vis.ipynb"]:::tool
    
    station_out --> fennil
    segment_out --> fennil
    meshes_csv --> fennil
    model_hdf --> parsli
    output_zarr --> vis_ipynb

    %% Vertical ordering trick    
    mesh_n ~~~ celeri_solve
    celeri_solve ~~~ station_out

    classDef tool fill:#e8d0e8,stroke:#333,stroke-width:1px
    classDef dashedtool fill:#e8d0e8,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    classDef dashed fill:#90EE90,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    classDef gradient fill:#90EE90,stroke:#333,stroke-width:1px
```

### Contributing to `celeri`

In order to contribute a GitHub pull request, you'll need to:

1. Set up your Python environment for development
2. Set up your local Git clone for contributions

If you've installed using pixi as described [above](#installation), your environment is already development-ready. Otherwise, you'll need to ensure `celeri` has been pip-installed in editable mode as per the [alternative installation instructions](alternative-installation.md#installing-for-development).

To reconfigure your local Git clone for contributions, you'll need to [fork the repository](https://github.com/brendanjmeade/celeri/fork) and reconfigure your remotes:

```bash
# After forking the repository to your GitHub account:
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/celeri.git
```

Replace `YOUR_USERNAME` with your GitHub username.

## Maintenance notes

See [maintenance-notes.md](maintenance-notes.md) for current best practices for maintaining this repository, in particular:

- [Cutting a new release](maintenance-notes.md#cutting-a-new-release)
- [Updating dependencies](maintenance-notes.md#updating-existing-dependencies)

## Other earthquake cycle kinematics software

We think celeri is pretty great, but there are other great kinematic modeling tools:

- Jack Loveless' and Brendan Meade's MATLAB-based [Blocks](https://github.com/jploveless/Blocks)
- Rob McCaffrey's Fortran-based [TDEFNODE](https://robmccaffrey.github.io/TDEFNODE/TDEFNODE.html)
- Richard Styron's Julia-based [Oiler](https://github.com/cossatot/Oiler)
