<p align="center">
  <img src="https://user-images.githubusercontent.com/4225359/132613223-257e6e17-83bd-49a4-8bbc-326cc117f6ec.png" width=400 />
</p>

## celeri - Next generation earthquake cycle and surface deformation modeling
A python port, reworking, and extension of the Matlab-based [blocks](https://github.com/jploveless/Blocks) featuring:
- Much smaller memory footprint
- Much faster elastic calculations
- Much faster block closure
- Eigenfunction expansion for partial coupling

To set up a development conda environment, run the following commands in the `celeri` folder.
```
conda config --prepend channels conda-forge
conda env create
conda activate celeri
pip install --no-use-pep517 -e .
```

Then start your favorite notebook viewer (`jupyter lab` or `vscode`) and open and run `celeri.ipynb`.

### Relationships of input files
```mermaid
flowchart TB
    c1<-->a2
    subgraph input_files
    a1<-->a2
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
    input_files --> two
    three --> two
    two --> c2
```

```mermaid
flowchart TD
  subgraph input_files
    command.json --> segment.csv
    command.json --> block.csv
    command.json --> station.csv
    command.json --> mesh_parameters.json
    mesh_parameters.json --> mesh_1.msh
    mesh_parameters.json --> mesh_2.msh
    mesh_parameters.json --> mesh_N.msh
    command.json --> elastic_precomputed.hdf5
  end
  subgraph celeri
    notebook.ipynb
  end
  subgraph output_files
    model_segment.csv
    model_block.csv
    model_velocties.csv
    elastic.hdf5
  end
  input_files --> celeri
  celeri --> output_files
```
