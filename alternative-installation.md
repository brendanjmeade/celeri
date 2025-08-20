# Installing `celeri`

There are many possible ways to install `celeri`. The choice depends on:

1. Whether you're developing `celeri` or not.
2. Which tool you prefer for managing your Python environment.

## Installing for development

If you're developing `celeri`, it's highly recommended to use [pixi](https://pixi.sh/) following the procedure in the [README](README.md). This will install `celeri` in a development environment with all dependencies.

If instead installing `celeri` with `conda`, `mamba`, `pip`, or `uv`, the simplest way to achieve a development environment is to first install the `celeri` package as per the [standalone instructions](#standalone-usage). This will ensure that all necessary dependencies are installed.

To prepare a standalone installation for development, run the following commands in your activated environment:

```bash
cd <path to your clone of this repository>
pip install --editable .
```

## Standalone usage

There are two main sources for the `celeri` package: conda-forge and PyPI. Depending on the package, the following tools can be used to install `celeri`:

- conda-forge (recommended):
  - [pixi](#pixi) (recommended, project-centric)
  - [conda](#conda-or-mamba) (creates global environments)
  - [mamba](#conda-or-mamba) or [micromamba](#conda-or-mamba) (fast conda substitutes)
- PyPI:
  - [pip](#pip)
  - [uv](#uv) (fast modern pip substitute)

### Conda-forge (recommended)

In order to provide the most robust experience for installing `celeri` and its dependencies, we recommend installing the [conda-forge package](https://prefix.dev/channels/conda-forge/packages/celeri).

### Pixi

[Pixi](https://pixi.sh/) is a flexible tool for managing conda-based projects. Please refer to the [README](README.md) for typical installation and usage.

#### Pixi with global command-line installation

If you just want to use `celeri` from the command line, you can install it globally:

```bash
pixi global install celeri
```

Then the `celeri-*` commands can be used without first running `pixi shell`.

#### Pixi with a different project

If you want to use `celeri` in a project that's not a clone of this repository, then you must initialize the project directory for pixi and install `celeri`:

```bash
cd my_project
pixi init        # Initialize the directory as an empty pixi project
pixi add celeri  # Add celeri as a project dependency
pixi shell       # Activate the project environment (triggers installation of celeri)
```

Inside this activated shell, you can run `celeri-solve` and other commands as usual.

After running `pixi shell`, you will be able to select the project's "default" environment in VS Code.

### Conda or mamba

With [conda](https://docs.anaconda.com/miniconda/) or [mamba](https://mamba.readthedocs.io/en/latest/), you can create a new environment with `celeri` and its dependencies:

```bash
conda config --prepend channels conda-forge  # Add conda-forge to the top of the channel list
conda create --name celeri celeri  # Create a new environment named "celeri" containing the "celeri" package
conda activate celeri
```

Note that `conda` can be substituted with `mamba` or `micromamba` for faster installation.

## PyPI-based

### pip

Avoid installing `celeri` in the global system environment since this can lead to dependency conflicts. It's recommended to install `celeri` into an isolated virtual environment:

```bash
python3 -m venv celeri  # Create a new virtual environment named celeri
source celeri/bin/activate
pip install celeri
```

### uv

You can use [uv](https://docs.astral.sh/uv/) for a more streamlined experience. It supports

- a project-centric approach similar to pixi (see the [uv documentation](https://docs.astral.sh/uv/project-management/))
- virtual environments

  ```bash
  uv venv celeri  # Create a new virtual environment named celeri
  source celeri/bin/activate
  uv pip install celeri
  ```

- global installation of the command-line tools

  ```bash
  uvx install celeri
  ```

- temporary environments for running `celeri` commands

  ```bash
  uvx --from celeri celeri-solve some_config.json
  ```
