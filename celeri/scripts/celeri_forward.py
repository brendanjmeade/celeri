#!/usr/bin/env python3

import argparse
import platform
import re
import subprocess
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from celeri.solve import Estimation


def is_m4_mac() -> bool:
    """True iff running on macOS and the CPU brand string contains 'M4'."""
    if platform.system() != "Darwin":
        return False
    out = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    )
    return "M4" in out


# One regex to match the three bogus matmul warnings
_MATMUL_MSG = r"(divide by zero|overflow|invalid value) encountered in matmul"


def silence_bogus_matmul_warnings() -> None:
    """Silence NumPy's spurious matmul RuntimeWarnings (see numpy#29820).

    This installs a warnings filter that ignores RuntimeWarnings whose
    message matches the bogus '... encountered in matmul' pattern.
    """
    warnings.filterwarnings(
        action="ignore",
        message=_MATMUL_MSG,
        category=RuntimeWarning,
    )

    # Filter out the dot product warnings from numpy.linalg
    warnings.filterwarnings(
        action="ignore",
        message=r"(divide by zero|overflow|invalid value) encountered in dot",
        category=RuntimeWarning,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create forward model predictions based on a previously run constrained model"
    )
    parser.add_argument(
        "model_run_dir",
        type=str,
        help="Path to the folder containing a previous model run",
    )
    parser.add_argument(
        "station_file",
        type=str,
        help="Path to a *_station.csv file containing longitude and latitude coordinates",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of stations to process in each batch (default: 1000)",
    )
    return parser.parse_args()


def load_station_coordinates(station_file_path):
    """Load longitude and latitude coordinates from a station CSV file."""
    df = pd.read_csv(station_file_path)

    # Check for required columns
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("Station file must contain 'lon' and 'lat' columns")

    return df


def create_forward_operators_batch(estimation, lon_batch, lat_batch):
    """Create operators for a batch of stations."""
    model = estimation.model
    operators = estimation.operators
    index = operators.index

    # Import the assign_block_labels function
    from celeri.model import assign_block_labels

    # Create temporary station dataframe for this batch
    batch_station = pd.DataFrame(
        {
            "lon": lon_batch,
            "lat": lat_batch,
            "east_vel": np.zeros(len(lon_batch)),
            "north_vel": np.zeros(len(lon_batch)),
            "east_sig": np.ones(len(lon_batch)),
            "north_sig": np.ones(len(lon_batch)),
            "flag": np.zeros(len(lon_batch), dtype=int),
            "up_vel": np.zeros(len(lon_batch)),
            "up_sig": np.ones(len(lon_batch)),
            "east_adjust": np.zeros(len(lon_batch)),
            "north_adjust": np.zeros(len(lon_batch)),
            "up_adjust": np.zeros(len(lon_batch)),
            "name": [f"FORWARD_{i}" for i in range(len(lon_batch))],
        }
    )

    # Create empty SAR dataframe (required for assign_block_labels)
    sar = pd.DataFrame()

    # Assign block labels to the batch stations
    closure, segment, batch_station, block, mogi, sar = assign_block_labels(
        model.segment, batch_station, model.block, model.mogi, sar
    )

    # Import the spatial functions we need
    from celeri.celeri_util import get_keep_index_12
    from celeri.spatial import (
        get_block_strain_rate_to_velocities_partials,
        get_mogi_to_velocities_partials,
        get_rotation_to_velocities_partials,
        get_segment_station_operator_okada,
        get_tde_to_velocities_single_mesh,
    )

    n_stations_batch = len(lon_batch)

    # Build rotation to velocities operator for this batch
    rotation_to_velocities_batch = get_rotation_to_velocities_partials(
        batch_station, len(model.block)
    )

    # Build elastic segment operator for this batch
    slip_rate_to_okada_to_velocities_batch = get_segment_station_operator_okada(
        model.segment, batch_station, model.config
    )

    # Combine rotation and elastic operators
    rotation_to_slip_rate_to_okada_to_velocities_batch = (
        slip_rate_to_okada_to_velocities_batch @ operators.rotation_to_slip_rate
    )

    # Build block strain rate operator if needed
    if index.n_strain_blocks > 0:
        block_strain_rate_to_velocities_batch, _ = (
            get_block_strain_rate_to_velocities_partials(block, batch_station, segment)
        )
        # Keep only the 2D components (remove vertical)
        keep_idx = get_keep_index_12(block_strain_rate_to_velocities_batch.shape[0])
        block_strain_rate_to_velocities_batch = block_strain_rate_to_velocities_batch[
            keep_idx, :
        ]
    else:
        block_strain_rate_to_velocities_batch = np.zeros((2 * n_stations_batch, 0))

    # Build Mogi operator if needed
    if index.n_mogis > 0:
        mogi_to_velocities_batch = get_mogi_to_velocities_partials(
            mogi, batch_station, model.config
        )
        # Keep only the 2D components (remove vertical)
        keep_idx = get_keep_index_12(mogi_to_velocities_batch.shape[0])
        mogi_to_velocities_batch = mogi_to_velocities_batch[keep_idx, :]
    else:
        mogi_to_velocities_batch = np.zeros((2 * n_stations_batch, 0))

    # Build TDE operators if needed
    tde_to_velocities_batch = None
    if operators.tde is not None and index.tde is not None:
        tde_to_velocities_batch = []
        for mesh_idx in range(len(model.meshes)):
            tde_op = get_tde_to_velocities_single_mesh(
                model.meshes, batch_station, model.config, mesh_idx
            )
            tde_to_velocities_batch.append(tde_op)

    return {
        "rotation_to_velocities": rotation_to_velocities_batch,
        "rotation_to_slip_rate_to_okada_to_velocities": rotation_to_slip_rate_to_okada_to_velocities_batch,
        "block_strain_rate_to_velocities": block_strain_rate_to_velocities_batch,
        "mogi_to_velocities": mogi_to_velocities_batch,
        "tde_to_velocities": tde_to_velocities_batch,
    }


def compute_forward_velocities_batch(estimation, batch_operators):
    """Compute forward velocities for a batch using the state vector and operators."""
    from celeri.celeri_util import get_keep_index_12

    state_vector = estimation.state_vector
    index = estimation.index

    # Extract rotation vector (first 3*n_blocks elements of state vector)
    rotation_vector = state_vector[0 : 3 * index.n_blocks]

    # Compute rotation velocities (3 components: east, north, up)
    vel_rotation_3d = batch_operators["rotation_to_velocities"] @ rotation_vector

    # Extract only horizontal components (east, north)
    keep_idx = get_keep_index_12(len(vel_rotation_3d))
    vel_rotation = vel_rotation_3d[keep_idx]

    # Get the number of stations in this batch (vel_rotation now has 2 components per station)
    n_stations_batch = len(vel_rotation) // 2

    # Compute elastic segment velocities (3 components: east, north, up)
    vel_elastic_segment_3d = (
        batch_operators["rotation_to_slip_rate_to_okada_to_velocities"]
        @ rotation_vector
    )

    # Extract only horizontal components (east, north)
    vel_elastic_segment = vel_elastic_segment_3d[keep_idx]

    # Compute block strain rate velocities
    if index.n_strain_blocks > 0:
        block_strain_rates = state_vector[
            index.start_block_strain_col : index.end_block_strain_col
        ]
        vel_block_strain_rate = (
            batch_operators["block_strain_rate_to_velocities"] @ block_strain_rates
        )
    else:
        vel_block_strain_rate = np.zeros(2 * n_stations_batch)

    # Compute Mogi velocities
    if index.n_mogis > 0:
        mogi_volume_change_rates = state_vector[
            index.start_mogi_col : index.end_mogi_col
        ]
        vel_mogi = batch_operators["mogi_to_velocities"] @ mogi_volume_change_rates
    else:
        vel_mogi = np.zeros(2 * n_stations_batch)

    # Compute TDE velocities
    if batch_operators["tde_to_velocities"] is not None and index.tde is not None:
        vel_tde = np.zeros(2 * n_stations_batch)

        if index.eigen is None:
            # Direct TDE mode
            for mesh_idx in range(len(batch_operators["tde_to_velocities"])):
                tde_op = batch_operators["tde_to_velocities"][mesh_idx]
                # Extract only horizontal components (remove vertical)
                tde_keep_row_index = get_keep_index_12(tde_op.shape[0])
                # TDE slip is stored as [strike_slip, dip_slip] pairs, so we need the 2-component index
                tde_keep_col_index = get_keep_index_12(tde_op.shape[1])

                tde_contribution = (
                    tde_op[tde_keep_row_index, :][:, tde_keep_col_index]
                    @ state_vector[
                        index.tde.start_tde_col[mesh_idx] : index.tde.end_tde_col[
                            mesh_idx
                        ]
                    ]
                )

                # Verify the contribution has the right shape
                if len(tde_contribution) != 2 * n_stations_batch:
                    raise ValueError(
                        f"TDE contribution has wrong shape: {tde_contribution.shape} vs expected {(2 * n_stations_batch,)}"
                    )

                vel_tde += tde_contribution
        else:
            # Eigenmode TDE
            assert estimation.operators.eigen is not None
            for mesh_idx in range(len(batch_operators["tde_to_velocities"])):
                tde_op = batch_operators["tde_to_velocities"][mesh_idx]
                eigenvectors = estimation.operators.eigen.eigenvectors_to_tde_slip[
                    mesh_idx
                ]

                # Combine TDE operator with eigenvectors
                # Extract only horizontal components (remove vertical)
                tde_keep_row_index = get_keep_index_12(tde_op.shape[0])
                tde_keep_col_index = get_keep_index_12(tde_op.shape[1])

                eigen_to_vel = -(
                    tde_op[tde_keep_row_index, :][:, tde_keep_col_index] @ eigenvectors
                )

                tde_contribution = (
                    eigen_to_vel
                    @ state_vector[
                        index.eigen.start_col_eigen[
                            mesh_idx
                        ] : index.eigen.end_col_eigen[mesh_idx]
                    ]
                )

                # Verify the contribution has the right shape
                if len(tde_contribution) != 2 * n_stations_batch:
                    raise ValueError(
                        f"Eigen TDE contribution has wrong shape: {tde_contribution.shape} vs expected {(2 * n_stations_batch,)}"
                    )

                vel_tde += tde_contribution
    else:
        vel_tde = np.zeros(2 * n_stations_batch)

    # Compute total velocities
    vel_total = vel_rotation + vel_elastic_segment

    # Compute residuals (forward model - observed, but observed is zero for forward stations)
    vel_residual = vel_total

    # Extract east and north components
    results = {
        "model_east_vel": vel_total[0::2],
        "model_north_vel": vel_total[1::2],
        "model_east_vel_residual": vel_residual[0::2],
        "model_north_vel_residual": vel_residual[1::2],
        "model_east_vel_rotation": vel_rotation[0::2],
        "model_north_vel_rotation": vel_rotation[1::2],
        "model_east_elastic_segment": vel_elastic_segment[0::2],
        "model_north_elastic_segment": vel_elastic_segment[1::2],
        "model_east_vel_tde": vel_tde[0::2],
        "model_north_vel_tde": vel_tde[1::2],
        "model_east_vel_block_strain_rate": vel_block_strain_rate[0::2],
        "model_north_vel_block_strain_rate": vel_block_strain_rate[1::2],
        "model_east_vel_mogi": vel_mogi[0::2],
        "model_north_vel_mogi": vel_mogi[1::2],
    }

    return results


def main():
    # HACK: Silence rogue M4 numpy warnings
    if is_m4_mac():
        silence_bogus_matmul_warnings()

    args = parse_args()

    # Extract the run folder name (characters after the last '/')
    run_folder_path = Path(args.model_run_dir)
    run_folder_name = run_folder_path.name

    # Extract the station filename without extension
    station_file_path = Path(args.station_file)
    station_file_base = station_file_path.stem  # Gets filename without extension

    # Generate UUID without dashes
    uuid_str = str(uuid.uuid4()).replace("-", "")

    # Create output filename with three parts: UUID_runfolder_stationfile.csv
    output_filename = f"{uuid_str}_{run_folder_name}_{station_file_base}.csv"
    logger.info(f"Output will be saved to: {output_filename}")

    # Load the estimation from the previous run
    logger.info(f"Loading estimation from: {args.model_run_dir}")
    estimation = Estimation.from_disk(args.model_run_dir)

    # Load the station coordinates
    logger.info(f"Loading station coordinates from: {args.station_file}")
    station_df = load_station_coordinates(args.station_file)

    n_stations = len(station_df)
    logger.info(f"Processing {n_stations} stations in batches of {args.batch_size}")

    # Initialize arrays to store all results
    all_results = {
        "lon": [],
        "lat": [],
        "model_east_vel": [],
        "model_north_vel": [],
        "model_east_vel_residual": [],
        "model_north_vel_residual": [],
        "model_east_vel_rotation": [],
        "model_north_vel_rotation": [],
        "model_east_elastic_segment": [],
        "model_north_elastic_segment": [],
        "model_east_vel_tde": [],
        "model_north_vel_tde": [],
        "model_east_vel_block_strain_rate": [],
        "model_north_vel_block_strain_rate": [],
        "model_east_vel_mogi": [],
        "model_north_vel_mogi": [],
    }

    # Process stations in batches
    for batch_start in range(0, n_stations, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_stations)
        batch_df = station_df.iloc[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_start // args.batch_size + 1}: stations {batch_start} to {batch_end}"
        )

        # Extract coordinates for this batch
        lon_batch = batch_df["lon"].values
        lat_batch = batch_df["lat"].values

        # Create operators for this batch
        batch_operators = create_forward_operators_batch(
            estimation, lon_batch, lat_batch
        )

        # Compute forward velocities for this batch
        batch_results = compute_forward_velocities_batch(estimation, batch_operators)

        # Append results
        all_results["lon"].extend(lon_batch)
        all_results["lat"].extend(lat_batch)
        for key in batch_results:
            all_results[key].extend(batch_results[key])

    # Create output DataFrame
    output_df = pd.DataFrame(all_results)

    # Add other columns that might be in the original station file but with default values
    if "name" in station_df.columns:
        output_df["name"] = station_df["name"].values
    else:
        output_df["name"] = [f"FORWARD_{i}" for i in range(n_stations)]

    # Add placeholder columns for compatibility with standard station format
    output_df["east_vel"] = 0.0
    output_df["north_vel"] = 0.0
    output_df["east_sig"] = 1.0
    output_df["north_sig"] = 1.0
    output_df["flag"] = 0
    output_df["up_vel"] = 0.0
    output_df["up_sig"] = 1.0
    output_df["east_adjust"] = 0.0
    output_df["north_adjust"] = 0.0
    output_df["up_adjust"] = 0.0
    output_df["corr"] = 0.0
    output_df["other1"] = 0

    # Reorder columns to match expected format
    column_order = [
        "lon",
        "lat",
        "corr",
        "other1",
        "name",
        "east_vel",
        "north_vel",
        "east_sig",
        "north_sig",
        "flag",
        "up_vel",
        "up_sig",
        "east_adjust",
        "north_adjust",
        "up_adjust",
        "model_east_vel",
        "model_north_vel",
        "model_east_vel_residual",
        "model_north_vel_residual",
        "model_east_vel_rotation",
        "model_north_vel_rotation",
        "model_east_elastic_segment",
        "model_north_elastic_segment",
        "model_east_vel_tde",
        "model_north_vel_tde",
        "model_east_vel_block_strain_rate",
        "model_north_vel_block_strain_rate",
        "model_east_vel_mogi",
        "model_north_vel_mogi",
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in output_df.columns]
    output_df = output_df[column_order]

    # Save to CSV
    output_df.to_csv(output_filename, index=False, float_format="%0.4f")
    logger.info(f"Forward model predictions saved to: {output_filename}")
    logger.info(f"Processed {n_stations} stations successfully")


if __name__ == "__main__":
    main()
