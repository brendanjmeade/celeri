from __future__ import annotations

import json
import os
import pickle
import shutil
import typing
from dataclasses import asdict

import h5py
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from celeri.config import Config

if typing.TYPE_CHECKING:
    from celeri.solve import Estimation


# TODO(Brendan): Why is there a pytest mark here?
@pytest.mark.skip(reason="Writing output files")
def write_output(
    config: Config,
    estimation: Estimation,
    station: pd.DataFrame,
    segment: pd.DataFrame,
    block: pd.DataFrame,
    meshes: dict,
    mogi=None,
):
    # Add model velocities to station dataframe and write .csv
    station["model_east_vel"] = estimation.east_vel
    station["model_north_vel"] = estimation.north_vel
    station["model_east_vel_residual"] = estimation.east_vel_residual
    station["model_north_vel_residual"] = estimation.north_vel_residual
    station["model_east_vel_rotation"] = estimation.east_vel_rotation
    station["model_north_vel_rotation"] = estimation.north_vel_rotation
    station["model_east_elastic_segment"] = estimation.east_vel_elastic_segment
    station["model_north_elastic_segment"] = estimation.north_vel_elastic_segment
    if config.solve_type != "dense_no_meshes":
        station["model_east_vel_tde"] = estimation.east_vel_tde
        station["model_north_vel_tde"] = estimation.north_vel_tde
    station["model_east_vel_block_strain_rate"] = estimation.east_vel_block_strain_rate
    station["model_north_vel_block_strain_rate"] = (
        estimation.north_vel_block_strain_rate
    )
    station["model_east_vel_mogi"] = estimation.east_vel_mogi
    station["model_north_vel_mogi"] = estimation.north_vel_mogi
    station_output_file_name = config.output_path + "/" + "model_station.csv"
    station.to_csv(station_output_file_name, index=False, float_format="%0.4f")

    # Add estimated slip rates to segment dataframe and write .csv
    segment["model_strike_slip_rate"] = estimation.strike_slip_rates
    segment["model_dip_slip_rate"] = estimation.dip_slip_rates
    segment["model_tensile_slip_rate"] = estimation.tensile_slip_rates
    segment["model_strike_slip_rate_uncertainty"] = (
        np.nan
        if estimation.strike_slip_rate_sigma is None
        else estimation.strike_slip_rate_sigma
    )
    segment["model_dip_slip_rate_uncertainty"] = (
        np.nan
        if estimation.dip_slip_rate_sigma is None
        else estimation.dip_slip_rate_sigma
    )
    segment["model_tensile_slip_rate_uncertainty"] = (
        np.nan
        if estimation.tensile_slip_rate_sigma is None
        else estimation.tensile_slip_rate_sigma
    )
    segment_output_file_name = config.output_path + "/" + "model_segment.csv"
    segment.to_csv(segment_output_file_name, index=False, float_format="%0.4f")

    # TODO: Add rotation rates and block strain rate block dataframe and write .csv
    block["euler_lon"] = estimation.euler_lon
    block["euler_lon_err"] = estimation.euler_lon_err
    block["euler_lat"] = estimation.euler_lat
    block["euler_lat_err"] = estimation.euler_lat_err
    block["euler_rate"] = estimation.euler_rate
    block["euler_rate_err"] = estimation.euler_rate_err
    block_output_file_name = config.output_path + "/" + "model_block.csv"
    block.to_csv(block_output_file_name, index=False, float_format="%0.4f")

    # Add volume change rates to Mogi source dataframe
    # Create an empy mogi dictionary if there isn't already one
    if mogi is None:
        mogi = []
    # mogi["volume_change"] = estimation.mogi_volume_change_rates
    # mogi["volume_change_sig"] = estimation.mogi_volume_change_rates

    # Construct mesh geometry dataframe
    if config.solve_type != "dense_no_meshes":
        mesh_outputs = pd.DataFrame()
        for i in range(len(meshes)):
            this_mesh_output = {
                "lon1": meshes[i].lon1,
                "lat1": meshes[i].lat1,
                "dep1": meshes[i].dep1,
                "lon2": meshes[i].lon2,
                "lat2": meshes[i].lat2,
                "dep2": meshes[i].dep2,
                "lon3": meshes[i].lon3,
                "lat3": meshes[i].lat3,
                "dep3": meshes[i].dep3,
            }
            this_mesh_output = pd.DataFrame(this_mesh_output)
            # mesh_outputs = mesh_outputs.append(this_mesh_output)
            mesh_outputs = pd.concat([mesh_outputs, this_mesh_output])

        # Append slip rates
        mesh_outputs["strike_slip_rate"] = estimation.tde_strike_slip_rates
        mesh_outputs["dip_slip_rate"] = estimation.tde_dip_slip_rates

        # Write to CSV
        mesh_output_file_name = config.output_path + "/" + "model_meshes.csv"
        mesh_outputs.to_csv(mesh_output_file_name, index=False)

        # Write a lot to a single hdf file
        def add_dataset(output_file_name, dataset_name, dataset):
            with h5py.File(output_file_name, "r+") as hdf:
                # Handle the case where dataset_name starts with '/'
                parts = dataset_name.strip("/").split("/")
                current_path = ""

                # Create groups for each level of the path
                for part in parts[:-1]:
                    if not part:  # Skip empty parts
                        continue

                    if current_path:
                        current_path = current_path + "/" + part
                    else:
                        current_path = part

                    # If this path exists and is a Dataset, delete it
                    if current_path in hdf and isinstance(
                        hdf[current_path], h5py.Dataset
                    ):
                        del hdf[current_path]

                    # Create group if it doesn't exist
                    if current_path not in hdf:
                        try:
                            hdf.create_group(current_path)
                        except ValueError:
                            raise

                # Now handle the final dataset
                if dataset_name in hdf:
                    del hdf[dataset_name]  # Remove the existing dataset

                # Create the new dataset
                hdf.create_dataset(dataset_name, data=dataset)

        hdf_output_file_name = (
            config.output_path + "/" + f"model_{config.run_name}.hdf5"
        )
        with h5py.File(hdf_output_file_name, "w") as hdf:
            # Meta data
            hdf.create_dataset(
                "run_name",
                data=config.run_name.encode("utf-8"),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            hdf.create_dataset("earth_radius", data=6371.0)

            # Write config dictionary
            grp = hdf.create_group("config")
            data = asdict(config)
            mesh_params = data.pop("mesh_params")
            for key, value in data.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    # Handle strings specially
                    grp.create_dataset(
                        key,
                        data=value.encode("utf-8"),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                else:
                    # Handle numeric values
                    grp.create_dataset(key, data=value)
            # Write mesh parameters
            for mesh_idx, data in enumerate(mesh_params):
                mesh_grp = grp.create_group(f"mesh_params/mesh_{mesh_idx:05}")
                for key, value in data.items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        mesh_grp.create_dataset(
                            key,
                            data=value.encode("utf-8"),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )
                    else:
                        mesh_grp.create_dataset(key, data=value)

            # Write meshes
            for i in range(len(meshes)):
                grp = hdf.create_group(f"meshes/mesh_{i:05}")
                mesh_name = os.path.splitext(os.path.basename(meshes[i].file_name))[0]
                grp.create_dataset(
                    "mesh_name",
                    data=mesh_name.encode("utf-8"),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                # grp.create_dataset("n_time_steps", data=1)

                # Write mesh geometry
                grp.create_dataset("coordinates", data=meshes[i].points)
                grp.create_dataset("verts", data=meshes[i].verts)

                # Write mesh scalars (we'll add more later)
                if i == 0:
                    mesh_start_idx = 0
                    mesh_end_idx = meshes[i].n_tde
                else:
                    mesh_start_idx = mesh_end_idx
                    mesh_end_idx = mesh_start_idx + meshes[i].n_tde

                # Write that there is a single timestep for parsli visualzation compatability
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/n_time_steps",
                    data=1,
                )

                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/strike_slip/{0:012}",
                    data=estimation.tde_strike_slip_rates[mesh_start_idx:mesh_end_idx],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/dip_slip/{0:012}",
                    data=estimation.tde_dip_slip_rates[mesh_start_idx:mesh_end_idx],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/tensile_slip/{0:012}",
                    data=np.zeros_like(
                        estimation.tde_dip_slip_rates[mesh_start_idx:mesh_end_idx]
                    ),
                )

                # Kinematic slip rates
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/strike_slip_kinematic/{0:012}",
                    data=estimation.tde_strike_slip_rates_kinematic[
                        mesh_start_idx:mesh_end_idx
                    ],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/dip_slip_kinematic/{0:012}",
                    data=estimation.tde_dip_slip_rates_kinematic[
                        mesh_start_idx:mesh_end_idx
                    ],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/tensile_slip_kinematic/{0:012}",
                    data=np.zeros_like(
                        estimation.tde_dip_slip_rates_kinematic[
                            mesh_start_idx:mesh_end_idx
                        ]
                    ),
                )

                # Coupling rates
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/strike_slip_coupling/{0:012}",
                    data=estimation.tde_strike_slip_rates_coupling[
                        mesh_start_idx:mesh_end_idx
                    ],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/dip_slip_coupling/{0:012}",
                    data=estimation.tde_dip_slip_rates_coupling[
                        mesh_start_idx:mesh_end_idx
                    ],
                )
                grp.create_dataset(
                    f"/meshes/mesh_{i:05}/tensile_slip_coupling/{0:012}",
                    data=np.zeros_like(
                        estimation.tde_dip_slip_rates_coupling[
                            mesh_start_idx:mesh_end_idx
                        ]
                    ),
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

            # Save segment information
            segment_no_name = segment.drop("name", axis=1)
            # Store the segment data
            hdf.create_dataset("segment", data=segment_no_name.to_numpy())
            # Store the segment column as a separate dataset
            string_dtype = h5py.string_dtype(
                encoding="utf-8"
            )  # Variable-length UTF-8 strings
            hdf.create_dataset(
                "segment_names",
                data=segment["name"].to_numpy(dtype=object),
                dtype=string_dtype,
            )
            # Store the column names as attributes
            hdf.attrs["columns"] = np.array(
                segment_no_name.columns, dtype=h5py.string_dtype()
            )
            # Store the index as an attribute
            hdf.attrs["index"] = segment_no_name.index.to_numpy()

            # Save station information
            station_no_name = station.drop("name", axis=1)
            # Store the station data
            hdf.create_dataset("station", data=station_no_name.to_numpy())
            # Store the segment column as a separate dataset
            string_dtype = h5py.string_dtype(
                encoding="utf-8"
            )  # Variable-length UTF-8 strings
            hdf.create_dataset(
                "station_names",
                data=station["name"].to_numpy(dtype=object),
                dtype=string_dtype,
            )
            # Store the column names as attributes
            hdf.attrs["columns"] = np.array(
                station_no_name.columns, dtype=h5py.string_dtype()
            )
            # Store the index as an attribute
            # hdf.attrs["index"] = station_no_name.index.to_numpy()

    # Write the config dict to an a json file
    args_config_output_file_name = (
        config.output_path + "/args_" + os.path.basename(config.file_name)
    )
    with open(args_config_output_file_name, "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Write all major variables to .pkl file in output folder
    with open(os.path.join(config.output_path, "output" + ".pkl"), "wb") as f:
        pickle.dump([config, estimation, station, segment, block, meshes], f)


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
