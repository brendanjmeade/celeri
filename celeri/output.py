from __future__ import annotations

import json
import os
import typing
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
import zarr

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from celeri.solve import Estimation


def write_output(
    estimation: Estimation,
):
    config = estimation.model.config
    output_path = Path(config.output_path)
    estimation.to_disk(output_path)
    station = estimation.model.station
    segment = estimation.model.segment
    meshes = estimation.model.meshes

    hdf_output_file_name = config.output_path / f"model_{config.run_name}.hdf5"
    with h5py.File(hdf_output_file_name, "w") as hdf:
        hdf.create_dataset(
            "run_name",
            data=config.run_name.encode("utf-8"),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        hdf.create_dataset("earth_radius", data=6371.0)

        # Write config dictionary
        grp = hdf.create_group("config")
        data = config.model_dump()
        mesh_params = data.pop("mesh_params")
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, str):
                grp.create_dataset(
                    key,
                    data=value.encode("utf-8"),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            elif np.issubdtype(type(value), np.number):
                grp.create_dataset(key, data=value)
            else:
                continue

        for mesh_idx, mesh_config in enumerate(mesh_params):
            mesh_grp = grp.create_group(f"mesh_params/mesh_{mesh_idx:05}")
            mesh_data = {}
            for field_name in mesh_config.keys():
                mesh_data[field_name] = mesh_config[field_name]

            for key, value in mesh_data.items():
                print(key, value)
                if value is None:
                    continue
                if isinstance(value, str):
                    mesh_grp.create_dataset(
                        key,
                        data=value.encode("utf-8"),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                elif isinstance(value, Path):
                    mesh_grp.create_dataset(
                        key,
                        data=str(value).encode("utf-8"),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                elif isinstance(value, int | float | np.integer | np.floating):
                    mesh_grp.create_dataset(key, data=value)

        mesh_end_idx = 0
        for i in range(len(meshes)):
            grp = hdf.create_group(f"meshes/mesh_{i:05}")
            mesh_name = os.path.splitext(os.path.basename(meshes[i].file_name))[0]
            grp.create_dataset(
                "mesh_name",
                data=mesh_name.encode("utf-8"),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            # Write mesh geometry
            grp.create_dataset("coordinates", data=meshes[i].points)
            grp.create_dataset("verts", data=meshes[i].verts)

            if i == 0:
                mesh_start_idx = 0
                mesh_end_idx = meshes[i].n_tde
            else:
                mesh_start_idx = mesh_end_idx
                mesh_end_idx = mesh_start_idx + meshes[i].n_tde

            # Write that there is a single timestep for parsli visualization compatability
            grp.create_dataset(
                f"/meshes/mesh_{i:05}/n_time_steps",
                data=1,
            )

            if estimation.tde_rates is not None:
                tde_ss_rates = estimation.tde_strike_slip_rates
                tde_ds_rates = estimation.tde_dip_slip_rates
                if tde_ss_rates is not None and tde_ds_rates is not None:
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/strike_slip/{0:012}",
                        data=tde_ss_rates[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/dip_slip/{0:012}",
                        data=tde_ds_rates[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/tensile_slip/{0:012}",
                        data=np.zeros_like(tde_ds_rates[i]),
                    )

                tde_ss_kinematic = estimation.tde_strike_slip_rates_kinematic
                tde_ds_kinematic = estimation.tde_dip_slip_rates_kinematic
                if tde_ss_kinematic is not None and tde_ds_kinematic is not None:
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/strike_slip_kinematic/{0:012}",
                        data=tde_ss_kinematic[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/dip_slip_kinematic/{0:012}",
                        data=tde_ds_kinematic[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/tensile_slip_kinematic/{0:012}",
                        data=np.zeros_like(tde_ds_kinematic[i]),
                    )

                coupling_ss = estimation.tde_strike_slip_rates_coupling
                coupling_ds = estimation.tde_dip_slip_rates_coupling
                if coupling_ss is not None and coupling_ds is not None:
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/strike_slip_coupling/{0:012}",
                        data=coupling_ss[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/dip_slip_coupling/{0:012}",
                        data=coupling_ds[i],
                    )
                    grp.create_dataset(
                        f"/meshes/mesh_{i:05}/tensile_slip_coupling/{0:012}",
                        data=np.zeros_like(coupling_ds[i]),
                    )

        # Try saving segment rate data in parsli style
        hdf.create_dataset(
            f"/segments/strike_slip/{0:012}", data=estimation.strike_slip_rates
        )
        hdf.create_dataset(
            f"/segments/dip_slip/{0:012}", data=estimation.dip_slip_rates
        )
        hdf.create_dataset(
            f"/segments/tensile_slip/{0:012}", data=estimation.tensile_slip_rates
        )

        segment_no_name = segment.drop("name", axis=1)
        hdf.create_dataset("segment", data=segment_no_name.to_numpy())
        string_dtype = h5py.string_dtype(encoding="utf-8")

        hdf.create_dataset(
            "segment_names",
            data=segment["name"].to_numpy(dtype=object),
            dtype=string_dtype,
        )

        hdf.attrs["columns"] = np.array(
            segment_no_name.columns, dtype=h5py.string_dtype()
        )

        hdf.attrs["index"] = segment_no_name.index.to_numpy()

        station_no_name = station.drop("name", axis=1)

        hdf.create_dataset("station", data=station_no_name.to_numpy())

        string_dtype = h5py.string_dtype(encoding="utf-8")

        hdf.create_dataset(
            "station_names",
            data=station["name"].to_numpy(dtype=object),
            dtype=string_dtype,
        )

        hdf.attrs["columns"] = np.array(
            station_no_name.columns, dtype=h5py.string_dtype()
        )

    args_config_output_file_name = (
        config.output_path / f"config.json"
    )
    with open(args_config_output_file_name, "w") as f:
        f.write(config.model_dump_json(indent=4))

    # Write model estimates to CSV files for easy access
    kwargs = {"index": False, "float_format": "%0.4f"}
    estimation.station.to_csv(output_path / "model_station.csv", **kwargs)
    estimation.segment.to_csv(output_path / "model_segment.csv", **kwargs)
    estimation.block.to_csv(output_path / "model_block.csv", **kwargs)
    estimation.mogi.to_csv(output_path / "model_mogi.csv", **kwargs)
    # Construct mesh geometry dataframe
    mesh_estimates = estimation.mesh_estimate
    if mesh_estimates is not None:
        mesh_estimates.to_csv(output_path / "model_meshes.csv", index=False)


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
    data = {
        field.name: getattr(obj, field.name)
        for field in fields(obj)
        if field.name not in skip
    }
    store = zarr.open_group(str(data_file), mode="w")
    remaining = {}
    for name, value in data.items():
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
                array_store = group.create_array(
                    str(key), shape=val.shape, dtype=val.dtype
                )
                array_store[...] = val
        elif isinstance(value, np.number):
            remaining[name] = value.item()
        else:
            remaining[name] = value

    # Save remaining non-array data
    remaining_file = output_dir / "remaining.json"
    with remaining_file.open("w") as f:
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
    with remaining_file.open() as f:
        remaining_data = json.load(f)
    data.update(remaining_data)
    data.update(extra)

    return cls(**data)
