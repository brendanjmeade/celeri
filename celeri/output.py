from __future__ import annotations

import json
import os
import pickle
import shutil
import typing
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import zarr
from loguru import logger

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from celeri.solve import Estimation


def write_output(
    estimation: Estimation,
):
    config = estimation.model.config
    output_path = Path(config.output_path)

    estimation.to_disk(output_path)

    # Write model estimates to CSV files for easy access
    estimation.station.to_csv(output_path / "model_station.csv", index=False, float_format="%0.4f")
    estimation.segment.to_csv(output_path / "model_segment.csv", index=False, float_format="%0.4f")
    estimation.block.to_csv(output_path / "model_block.csv", index=False, float_format="%0.4f")
    estimation.mogi.to_csv(output_path / "model_mogi.csv", index=False, float_format="%0.4f")
    # Construct mesh geometry dataframe
    mesh_estimates = estimation.mesh_estimate
    if mesh_estimates is not None:
        mesh_estimates.to_csv(output_path / "model_meshes.csv", index=False)


def write_output_supplemental(
    args, config, index, data, operators, estimation, assembly
):
    # Copy all input files to output folder
    file_names = [
        "segment_file_name",
        "station_file_name",
        "block_file_name",
        "mesh_parameters_file_name",
        "los_file_name",
        "file_name",
    ]
    for file_name in file_names:
        try:
            shutil.copyfile(
                config[file_name],
                os.path.join(
                    config.output_path,
                    os.path.basename(os.path.normpath(config[file_name])),
                ),
            )
        except:
            logger.warning(f"No {file_name} to copy to output folder")

    # Copy .msh files to output foler
    if len(data.meshes) > 0:
        for i in range(len(data.meshes)):
            msh_file_name = data.meshes[i].file_name
            try:
                shutil.copyfile(
                    msh_file_name,
                    os.path.join(
                        config.output_path,
                        os.path.basename(os.path.normpath(msh_file_name)),
                    ),
                )
            except:
                logger.warning(f"No {msh_file_name} to copy to output folder")

    # Write config line arguments to output folder
    with open(
        os.path.join(config.output_path, config.run_name + "_args.json"), "w"
    ) as f:
        json.dump(args, f, indent=2)

    # Write all major variables to .pkl file in output folder
    if bool(config.pickle_save):
        with open(
            os.path.join(config.output_path, config.run_name + ".pkl"), "wb"
        ) as f:
            pickle.dump([config, index, data, operators, estimation, assembly], f)


def dataclass_to_disk(
    obj: DataclassInstance, output_dir: str | Path, *, skip: set[str] | None = None
):
    """Save a dataclass object to disk as JSON and Zarr.

    This function can handle dataclasses that contain numpy arrays
    and other data types that are json-serializable, like integers,
    floats or strings. It also allows dictionaries with integer keys
    that contain only numpy arrays.

    It does not handle nested dataclasses or complex objects directly,
    but you can handle those manually and pass the manually handeled
    attribute names to `skip`.
    """
    if skip is None:
        skip = set()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the arrays as zarr file
    data_file = output_dir / "arrays.zarr"
    data = asdict(obj)
    store = zarr.open_group(str(data_file), mode="w")
    remaining = {}
    for name, value in data.items():
        if name in skip:
            continue
        if isinstance(value, np.ndarray):
            array_store = store.create_array(name, shape=value.shape, dtype=value.dtype)
            array_store[...] = value
        elif (
            isinstance(value, dict)
            and all(isinstance(k, int) for k in value.keys())
            and all(isinstance(v, np.ndarray) for v in value.values())
        ):
            group = store.create_group(name)
            for key, val in value.items():
                array_store = group.create_array(str(key), shape=val.shape, dtype=val.dtype)
                array_store[...] = val
        elif isinstance(value, np.number):
            remaining[name] = value.item()
        else:
            remaining[name] = value

    # Save remaining non-array data
    remaining_file = output_dir / "remaining.json"
    with open(remaining_file, "w") as f:
        json.dump(remaining, f, indent=4)


T = TypeVar("T")


def dataclass_from_disk(
    cls: type[T], input_dir: str | Path, *, extra: dict[str, Any] | None = None
) -> T:
    """Load a dataclass object from disk.

    Args:
        cls: The dataclass type to instantiate
        input_dir: Directory containing the saved data
        extra: Additional attributes to add to the loaded data

    Returns:
        An instance of the dataclass
    """
    if extra is None:
        extra = {}

    input_dir = Path(input_dir)

    # Load the arrays from zarr file
    data_file = input_dir / "arrays.zarr"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found.")
    store = zarr.open_group(str(data_file), mode="r")
    data = {}
    for name in store.array_keys():
        if name in extra:
            raise ValueError(f"Duplicate key '{name}' found in extra data.")
        values = store[name]
        assert isinstance(values, zarr.Array)
        data[name] = values[...]
    for name in store.group_keys():
        values = store[name]
        assert isinstance(values, zarr.Group)
        arrays = {}
        for key in values.array_keys():
            array_values = values[key]
            if isinstance(array_values, zarr.Group):
                raise TypeError(
                    f"zarr file contains nested groups at key '{name}/{key}'"
                )
            arrays[int(key)] = array_values[...]
        data[name] = arrays

    # Load remaining non-array data
    remaining_file = input_dir / "remaining.json"
    if not remaining_file.exists():
        raise FileNotFoundError(f"Remaining data file {remaining_file} not found.")
    with open(remaining_file) as f:
        remaining_data = json.load(f)
    data.update(remaining_data)
    data.update(extra)

    return cls(**data)
