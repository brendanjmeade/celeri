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

### The flow of files in and out of celeri
Files with dark orange background shading are required (automatically generated) and those with light blue background shading are optional (not automatically generated)
```mermaid
flowchart TD
  classDef required fill:#f96;
  subgraph input_files
    command.json:::required --> segment.csv:::required
    command.json:::required --> block.csv:::required
    command.json:::required --> station.csv:::required
    subgraph meshes
      mesh_parameters.json
      mesh_parameters.json --> mesh_1.msh
      mesh_parameters.json --> mesh_2.msh
      mesh_parameters.json --> mesh_N.msh
    end
    command.json --> meshes
    command.json --> elastic_operators_precomputed.hdf5
    command.json --> los.csv
  end
  subgraph celeri
    notebook.ipynb:::required
  end
  subgraph output_files
    model_segment.csv:::required
    model_block.csv:::required
    elastic_operators.hdf5
    subgraph velocities
      model_velocity.csv:::required
      rotation_velocity.csv:::required
      strain_rate_velocity.csv:::required
      okada_velocity.csv:::required
      tde_velocity.csv:::required
      elastic_velocity.csv:::required
    end
    subgraph los
      model_los.csv
      rotation_los.csv
      strain_rate_los.csv
      okada_los.csv
      tde_los.csv
      elastic_los.csv
    end
  end
  input_files --> celeri
  celeri --> output_files
```
