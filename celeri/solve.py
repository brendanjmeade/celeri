# TODO figure out how to distinguish between different solvers
# Right now we have solve.py and optimize.py. This is a bit
# confusing.
from __future__ import annotations

import timeit
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import cvxopt
import numpy as np
import pandas as pd
import scipy
from loguru import logger
from scipy import linalg

from celeri.celeri_util import get_keep_index_12
from celeri.model import Model
from celeri.operators import (
    Index,
    Operators,
    _get_data_vector,
    _get_data_vector_eigen,
    _get_data_vector_no_meshes,
    _get_weighting_vector,
    _get_weighting_vector_eigen,
    _get_weighting_vector_no_meshes,
    build_operators,
    rotation_vector_err_to_euler_pole_err,
    rotation_vectors_to_euler_poles,
)
from celeri.output import dataclass_from_disk, dataclass_to_disk, write_output
from celeri.version import __version__ as celeri_version


@dataclass(kw_only=True)
class Estimation:
    """A class to hold an estimation of the model parameters."""

    data_vector: np.ndarray
    """The data vector, containing the data and constraints."""
    weighting_vector: np.ndarray
    """The weighting vector, containing the weights for the data and constraints."""
    state_vector: np.ndarray
    """The state vector, containing the model parameters."""
    operators: Operators
    """The operators object, containing the operators for the forward model."""
    state_covariance_matrix: np.ndarray | None
    """The covariance matrix of the state vector."""
    n_out_of_bounds_trace: np.ndarray | None = None
    """Trace of out-of-bounds values during optimization."""
    trace: Any | None = None
    """Trace from optimization."""
    mcmc_trace: Any | None = None
    """MCMC trace from Bayesian inference."""
    mcmc_start_time: str | None = None
    """Start time of MCMC sampling."""
    mcmc_end_time: str | None = None
    """End time of MCMC sampling."""
    mcmc_duration: float | None = None
    """Duration of MCMC sampling."""
    mcmc_num_divergences: int | None = None
    """Number of divergent transitions in MCMC sampling."""
    mcmc_dropped_chains: list[int] | None = None
    """Chain labels dropped by the stalled-chain watchdog.

    ``None`` when the watchdog was not enabled; an empty list when it ran
    and dropped nothing.
    """
    celeri_version: str | None = None
    """Version of celeri used to create this estimation."""

    @property
    def model(self) -> Model:
        """The model object."""
        return self.operators.model

    @property
    def index(self) -> Index:
        """The index object for the operators."""
        return self.operators.index

    @property
    def operator(self) -> np.ndarray:
        """The full operator, containing the linear operator for the forward model."""
        return self.operators.full_dense_operator

    @cached_property
    def station(self) -> pd.DataFrame:
        """An extension of the `model.station` dataframe, with additional columns
        for the estimated velocities returned by the model.
        """
        station = self.model.station.copy(deep=True)
        station["model_east_vel"] = self.east_vel
        station["model_north_vel"] = self.north_vel
        station["model_east_vel_residual"] = self.east_vel_residual
        station["model_north_vel_residual"] = self.north_vel_residual
        station["model_east_vel_rotation"] = self.east_vel_rotation
        station["model_north_vel_rotation"] = self.north_vel_rotation
        station["model_east_elastic_segment"] = self.east_vel_elastic_segment
        station["model_north_elastic_segment"] = self.north_vel_elastic_segment
        if self.east_vel_tde is not None:
            station["model_east_vel_tde"] = self.east_vel_tde
        if self.north_vel_tde is not None:
            station["model_north_vel_tde"] = self.north_vel_tde
        if self.up_vel_tde is not None:
            station["model_up_vel_tde"] = self.up_vel_tde
        station["model_east_vel_block_strain_rate"] = self.east_vel_block_strain_rate
        station["model_north_vel_block_strain_rate"] = self.north_vel_block_strain_rate
        station["model_east_vel_mogi"] = self.east_vel_mogi
        station["model_north_vel_mogi"] = self.north_vel_mogi
        station["model_up_vel"] = self.up_vel
        station["model_up_vel_residual"] = self.up_vel_residual
        station["model_up_vel_rotation"] = self.up_vel_rotation
        station["model_up_elastic_segment"] = self.up_vel_elastic_segment
        station["model_up_vel_block_strain_rate"] = self.up_vel_block_strain_rate
        station["model_up_vel_mogi"] = self.up_vel_mogi
        return station

    @cached_property
    def segment(self) -> pd.DataFrame:
        """An extension of the `model.segment` dataframe, with additional columns for the estimated slip rates returned by the model."""
        segment = self.model.segment.copy(deep=True)
        segment["model_strike_slip_rate"] = self.strike_slip_rates
        segment["model_dip_slip_rate"] = self.dip_slip_rates
        segment["model_tensile_slip_rate"] = self.tensile_slip_rates
        segment["model_strike_slip_rate_uncertainty"] = (
            np.nan
            if self.strike_slip_rate_sigma is None
            else self.strike_slip_rate_sigma
        )
        segment["model_dip_slip_rate_uncertainty"] = (
            np.nan if self.dip_slip_rate_sigma is None else self.dip_slip_rate_sigma
        )
        segment["model_tensile_slip_rate_uncertainty"] = (
            np.nan
            if self.tensile_slip_rate_sigma is None
            else self.tensile_slip_rate_sigma
        )
        return segment

    @cached_property
    def block(self) -> pd.DataFrame:
        """An extension of the `model.block` dataframe, with additional columns for the estimated Euler pole parameters returned by the model."""
        block = self.model.block.copy(deep=True)
        block["euler_lon"] = self.euler_lon
        block["euler_lon_err"] = self.euler_lon_err
        block["euler_lat"] = self.euler_lat
        block["euler_lat_err"] = self.euler_lat_err
        block["euler_rate"] = self.euler_rate
        block["euler_rate_err"] = self.euler_rate_err
        return block

    @cached_property
    def mogi(self) -> pd.DataFrame:
        """An extension of the `model.mogi` dataframe, with additional columns for the estimated mogi parameters returned by the model."""
        mogi = self.model.mogi.copy(deep=True)
        # TODO Why the different names?
        mogi["volume_change"] = self.mogi_volume_change_rates
        mogi["volume_change_sig"] = self.mogi_volume_change_rates
        mogi["volume_change_rates"] = self.mogi_volume_change_rates
        return mogi

    @cached_property
    def _mcmc_posterior_mean(self):
        """The MCMC posterior averaged over whatever sample dimensions it has."""
        assert self.mcmc_trace is not None
        posterior = self.mcmc_trace.posterior
        sample_dims = [dim for dim in ("chain", "draw") if dim in posterior.dims]
        return posterior.mean(sample_dims) if sample_dims else posterior

    def mesh_slip_fields(self, mesh_idx: int) -> dict[str, np.ndarray | None]:
        """The estimated slip fields of one mesh, as written to the output files.

        Returns the elastic slip rates, kinematic slip rates and couplings for
        strike slip and dip slip on mesh ``mesh_idx``. An entry is ``None``
        when it is not defined for this estimation (no TDEs, or kinematic
        rates for a mesh that is not tied to any segment). Both
        ``mesh_estimate`` (``model_meshes.csv``) and the HDF5 writer read
        from here so that the two outputs agree.

        For MCMC estimations the elastic rates and couplings are the
        posterior means of the sampled ``elastic_*`` and ``coupling_*``
        fields, and the kinematic rates are the unsmoothed rates the sampler
        used, evaluated at the posterior mean rotation (which is the
        posterior mean kinematic rate, since the map is linear). Note that
        ``tde_strike_slip_rates``/``tde_dip_slip_rates`` differ from these
        for MCMC: they hold the projection of the posterior mean elastic
        field onto the truncated eigenmode basis, i.e. the part of the field
        that the sampler's forward model sees. When coupling was not sampled
        (elastic-mode meshes) the coupling is the ratio elastic / kinematic.

        For other solvers the elastic rates come from the state vector, the
        kinematic rates are Gaussian smoothed when eigenmodes are in use,
        and the coupling is the ratio elastic / kinematic.
        """
        fields: dict[str, np.ndarray | None] = {
            "strike_slip_rate": None,
            "dip_slip_rate": None,
            "strike_slip_rate_kinematic": None,
            "dip_slip_rate_kinematic": None,
            "strike_slip_coupling": None,
            "dip_slip_coupling": None,
        }
        if self.operators.tde is None:
            return fields
        strike_slip_rates = self.tde_strike_slip_rates
        dip_slip_rates = self.tde_dip_slip_rates
        if strike_slip_rates is None or dip_slip_rates is None:
            return fields
        fields["strike_slip_rate"] = strike_slip_rates[mesh_idx]
        fields["dip_slip_rate"] = dip_slip_rates[mesh_idx]

        if self.mcmc_trace is not None:
            posterior_mean = self._mcmc_posterior_mean
            kinematic = self.tde_kinematic.get(mesh_idx, None)
            for kind, key, component in (
                ("ss", "strike_slip", 0),
                ("ds", "dip_slip", 1),
            ):
                elastic_var = f"elastic_{mesh_idx}_{kind}"
                if elastic_var in posterior_mean:
                    fields[f"{key}_rate"] = np.asarray(
                        posterior_mean[elastic_var].values, dtype=float
                    )
                if kinematic is not None:
                    fields[f"{key}_rate_kinematic"] = kinematic[component::2]
                coupling_var = f"coupling_{mesh_idx}_{kind}"
                if coupling_var in posterior_mean:
                    fields[f"{key}_coupling"] = np.asarray(
                        posterior_mean[coupling_var].values, dtype=float
                    )
                elif kinematic is not None:
                    rate = fields[f"{key}_rate"]
                    assert rate is not None
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fields[f"{key}_coupling"] = rate / kinematic[component::2]
            return fields

        if self.operators.eigen is not None:
            fields["strike_slip_rate_kinematic"] = (
                self.tde_strike_slip_rates_kinematic_smooth.get(mesh_idx, None)
            )
            fields["dip_slip_rate_kinematic"] = (
                self.tde_dip_slip_rates_kinematic_smooth.get(mesh_idx, None)
            )
            if (coupling := self.tde_strike_slip_rates_coupling_smooth) is not None:
                fields["strike_slip_coupling"] = coupling.get(mesh_idx, None)
            if (coupling := self.tde_dip_slip_rates_coupling_smooth) is not None:
                fields["dip_slip_coupling"] = coupling.get(mesh_idx, None)
        else:
            fields["strike_slip_rate_kinematic"] = (
                self.tde_strike_slip_rates_kinematic.get(mesh_idx, None)
            )
            fields["dip_slip_rate_kinematic"] = self.tde_dip_slip_rates_kinematic.get(
                mesh_idx, None
            )
            if (coupling := self.tde_strike_slip_rates_coupling) is not None:
                fields["strike_slip_coupling"] = coupling.get(mesh_idx, None)
            if (coupling := self.tde_dip_slip_rates_coupling) is not None:
                fields["dip_slip_coupling"] = coupling.get(mesh_idx, None)
        return fields

    @cached_property
    def mesh_estimate(self) -> pd.DataFrame | None:
        """A dataframe containing the estimated slip rates and couplings for each mesh.

        The slip columns come from `mesh_slip_fields`; fields that are not
        defined for a mesh are written as zeros.
        """
        if self.operators.tde is None:
            return None
        if self.tde_strike_slip_rates is None or self.tde_dip_slip_rates is None:
            return None
        meshes = self.model.meshes
        mesh_outputs_list: list[pd.DataFrame] = []
        for i in range(len(meshes)):
            n_elements = len(meshes[i].lon1)
            slip_fields = {
                key: np.zeros(n_elements) if value is None else value
                for key, value in self.mesh_slip_fields(i).items()
            }
            this_mesh_data = {
                "lon1": meshes[i].lon1,
                "lat1": meshes[i].lat1,
                "dep1": meshes[i].dep1,
                "lon2": meshes[i].lon2,
                "lat2": meshes[i].lat2,
                "dep2": meshes[i].dep2,
                "lon3": meshes[i].lon3,
                "lat3": meshes[i].lat3,
                "dep3": meshes[i].dep3,
                "mesh_idx": i * np.ones_like(meshes[i].lon1).astype(int),
                **slip_fields,
            }

            mesh_outputs_list.append(pd.DataFrame(this_mesh_data))

        # Concatenate all DataFrames at once, or return empty DataFrame if no meshes
        if mesh_outputs_list:
            mesh_outputs = pd.concat(mesh_outputs_list, ignore_index=True)
        else:
            mesh_outputs = pd.DataFrame()

        return mesh_outputs

    @cached_property
    def predictions(self) -> np.ndarray:
        """The full forward model predictions vector."""
        return self.operator @ self.state_vector

    @property
    def vel(self) -> np.ndarray:
        """The estimated velocities at the stations (always 3 components: east, north, up)."""
        return self.predictions[0 : 3 * self.index.n_stations]

    @property
    def east_vel(self) -> np.ndarray:
        """The estimated east velocities at the stations."""
        return self.vel[0::3]

    @property
    def north_vel(self) -> np.ndarray:
        """The estimated north velocities at the stations."""
        return self.vel[1::3]

    @property
    def up_vel(self) -> np.ndarray:
        """The estimated up velocities at the stations."""
        return self.vel[2::3]

    @property
    def east_vel_residual(self) -> np.ndarray:
        """The residual between the estimated and observed east velocities at the stations."""
        return self.east_vel - self.model.station.east_vel

    @property
    def north_vel_residual(self) -> np.ndarray:
        """The residual between the estimated and observed north velocities at the stations."""
        return self.north_vel - self.model.station.north_vel

    @property
    def up_vel_residual(self) -> np.ndarray:
        """The residual between the estimated and observed up velocities at the stations."""
        return self.up_vel - self.model.station.up_vel

    @property
    def rotation_vector(self) -> np.ndarray:
        """Returns an np.array of length 3 * n_blocks, containing (x,y,z) rotation vector components for each block."""
        return self.state_vector[0 : 3 * self.index.n_blocks]

    @property
    def rotation_vector_x(self) -> np.ndarray:
        """The estimated x-component of the rotation vector."""
        return self.rotation_vector[0::3]

    @property
    def rotation_vector_y(self) -> np.ndarray:
        """The estimated y-component of the rotation vector."""
        return self.rotation_vector[1::3]

    @property
    def rotation_vector_z(self) -> np.ndarray:
        """The estimated z-component of the rotation vector."""
        return self.rotation_vector[2::3]

    @cached_property
    def slip_rate_sigma(self) -> np.ndarray | None:
        """The standard deviation of the estimated strike, dip, and tensile slip rates for each segment. Shape: (3 * n_segments,).

        Propagated from `self.state_covariance_matrix` when it is available.
        For MCMC estimations it is the posterior standard deviation of the
        sampled ``segment_slip_rate`` variable.
        """
        if self.state_covariance_matrix is not None:
            return np.sqrt(
                np.diag(
                    self.operators.rotation_to_slip_rate
                    @ self.state_covariance_matrix[
                        0 : 3 * self.index.n_blocks, 0 : 3 * self.index.n_blocks
                    ]
                    @ self.operators.rotation_to_slip_rate.T
                )
            )
        if self.mcmc_trace is not None:
            posterior = self.mcmc_trace.posterior
            if "segment_slip_rate" in posterior:
                rates = posterior["segment_slip_rate"]
                sample_dims = [dim for dim in ("chain", "draw") if dim in rates.dims]
                if sample_dims:
                    sigma = rates.std(sample_dims).values
                else:
                    # A single draw: no spread to report
                    sigma = np.zeros(rates.shape)
                # (n_segments, 3) with components (strike, dip, tensile)
                # -> interleaved like `slip_rates`
                return np.asarray(sigma, dtype=float).ravel()
        return None

    @property
    def strike_slip_rate_sigma(self) -> np.ndarray | None:
        """The sigma of the strike slip rates."""
        if self.slip_rate_sigma is not None:
            return self.slip_rate_sigma[0::3]
        return None

    @property
    def dip_slip_rate_sigma(self) -> np.ndarray | None:
        """The sigma of the dip slip rates."""
        if self.slip_rate_sigma is not None:
            return self.slip_rate_sigma[1::3]
        return None

    @property
    def tensile_slip_rate_sigma(self) -> np.ndarray | None:
        """The sigma of the tensile slip rates."""
        if self.slip_rate_sigma is not None:
            return self.slip_rate_sigma[2::3]
        return None

    @cached_property
    def tde_rates(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to TDE slip rate arrays."""
        index = self.index
        if index.tde is None:
            return None

        if index.eigen is None:
            return {
                mesh_idx: self.state_vector[
                    index.tde.start_tde_col[mesh_idx] : index.tde.end_tde_col[mesh_idx]
                ]
                for mesh_idx in range(len(self.model.meshes))
            }

        assert self.operators.eigen is not None

        tde_rates = {}
        for mesh_idx in range(index.n_meshes):
            tde_rates[mesh_idx] = (
                self.operators.eigen.eigenvectors_to_tde_slip[mesh_idx]
                @ self.state_vector[
                    index.eigen.start_col_eigen[mesh_idx] : index.eigen.end_col_eigen[
                        mesh_idx
                    ]
                ]
            )
        return tde_rates

    @property
    def tde_strike_slip_rates(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to TDE strike slip rate arrays."""
        if (rates := self.tde_rates) is None:
            return None
        return {key: val[0::2] for key, val in rates.items()}

    @property
    def tde_dip_slip_rates(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to TDE dip slip rate arrays."""
        if (rates := self.tde_rates) is None:
            return None
        return {key: val[1::2] for key, val in rates.items()}

    @property
    def slip_rates(self) -> np.ndarray:
        """The estimated slip rates (strike, dip, tensile) for each segment."""
        return self.operators.rotation_to_slip_rate @ self.rotation_vector

    @property
    def strike_slip_rates(self) -> np.ndarray:
        """The estimated strike slip rates for each segment."""
        return self.slip_rates[0::3]

    @property
    def dip_slip_rates(self) -> np.ndarray:
        """The estimated dip slip rates for each segment."""
        return self.slip_rates[1::3]

    @property
    def tensile_slip_rates(self) -> np.ndarray:
        """The estimated tensile slip rates for each segment."""
        return self.slip_rates[2::3]

    @property
    def block_strain_rates(self) -> np.ndarray:
        """The estimated block strain rates."""
        # TODO(Adrian) verify with eigen
        return self.state_vector[
            self.index.start_block_strain_col : self.index.end_block_strain_col
        ]

    @property
    def mogi_volume_change_rates(self) -> np.ndarray:
        """The estimated Mogi volume change rates."""
        # TODO(Adrian) verify with eigen
        return self.state_vector[self.index.start_mogi_col : self.index.end_mogi_col]

    @property
    def vel_rotation(self) -> np.ndarray:
        """Returns an np.array of shape (n_stations,), containing the velocity components (unstacked) for each station."""
        return (
            self.operators.rotation_to_velocities[self.index.station_row_keep_index, :]
            @ self.rotation_vector
        )

    @property
    def east_vel_rotation(self) -> np.ndarray:
        """Returns an np.array of shape (n_stations,), containing the east velocity components for each station."""
        return self.vel_rotation[0::3]

    @property
    def north_vel_rotation(self) -> np.ndarray:
        """Returns an np.array of shape (n_stations,), containing the north velocity components for each station."""
        return self.vel_rotation[1::3]

    @property
    def up_vel_rotation(self) -> np.ndarray:
        """Returns an np.array of shape (n_stations,), containing the up velocity components for each station."""
        return self.vel_rotation[2::3]

    @cached_property
    def vel_elastic_segment(self) -> np.ndarray:
        """Elastic velocities on the segments from Okada."""
        return (
            self.operators.rotation_to_slip_rate_to_okada_to_velocities[
                self.index.station_row_keep_index, :
            ]
            @ self.rotation_vector
        )

    @property
    def east_vel_elastic_segment(self) -> np.ndarray:
        """East component of elastic velocities on the segments from Okada."""
        return self.vel_elastic_segment[0::3]

    @property
    def north_vel_elastic_segment(self) -> np.ndarray:
        """North component of elastic velocities on the segments from Okada."""
        return self.vel_elastic_segment[1::3]

    @property
    def up_vel_elastic_segment(self) -> np.ndarray:
        """Up component of elastic velocities on the segments from Okada."""
        return self.vel_elastic_segment[2::3]

    @cached_property
    def vel_block_strain_rate(self) -> np.ndarray:
        """Velocities from block strain rates."""
        return (
            self.operators.block_strain_rate_to_velocities[
                self.index.station_row_keep_index, :
            ]
            @ self.block_strain_rates
        )

    @property
    def east_vel_block_strain_rate(self) -> np.ndarray:
        """East component of velocities from block strain rates."""
        return self.vel_block_strain_rate[0::3]

    @property
    def north_vel_block_strain_rate(self) -> np.ndarray:
        """North component of velocities from block strain rates."""
        return self.vel_block_strain_rate[1::3]

    @property
    def up_vel_block_strain_rate(self) -> np.ndarray:
        """Up component of velocities from block strain rates."""
        return self.vel_block_strain_rate[2::3]

    @cached_property
    def euler(self) -> np.ndarray:
        """The estimated euler poles of rotation for each block. Shape: (3, n_blocks)."""
        lon, lat, rate = rotation_vectors_to_euler_poles(
            self.rotation_vector_x, self.rotation_vector_y, self.rotation_vector_z
        )
        return np.array([lon, lat, rate])

    @property
    def euler_lon(self) -> np.ndarray:
        """The estimated Euler pole longitude for each block."""
        return self.euler[0]

    @property
    def euler_lat(self) -> np.ndarray:
        """The estimated Euler pole latitude for each block."""
        return self.euler[1]

    @property
    def euler_rate(self) -> np.ndarray:
        """The estimated Euler pole rotation rate for each block."""
        return self.euler[2]

    @cached_property
    def euler_err(self) -> np.ndarray:
        """The estimated Euler pole errors (lon, lat, rate) for each block."""
        # TODO
        omega_cov = np.zeros(
            (3 * len(self.rotation_vector_x), 3 * len(self.rotation_vector_x))
        )
        lon_err, lat_err, rate_err = rotation_vector_err_to_euler_pole_err(
            self.rotation_vector_x,
            self.rotation_vector_y,
            self.rotation_vector_z,
            omega_cov,
        )
        return np.array([lon_err, lat_err, rate_err])

    @property
    def euler_lon_err(self) -> np.ndarray:
        """The estimated Euler pole longitude error for each block."""
        return self.euler_err[0]

    @property
    def euler_lat_err(self) -> np.ndarray:
        """The estimated Euler pole latitude error for each block."""
        return self.euler_err[1]

    @property
    def euler_rate_err(self) -> np.ndarray:
        """The estimated Euler pole rotation rate error for each block."""
        return self.euler_err[2]

    @cached_property
    def vel_mogi(self) -> np.ndarray:
        """Velocities from Mogi sources."""
        return (
            self.operators.mogi_to_velocities[self.index.station_row_keep_index, :]
            @ self.mogi_volume_change_rates
        )

    @property
    def east_vel_mogi(self) -> np.ndarray:
        """East component of velocities from Mogi sources."""
        return self.vel_mogi[0::3]

    @property
    def north_vel_mogi(self) -> np.ndarray:
        """North component of velocities from Mogi sources."""
        return self.vel_mogi[1::3]

    @property
    def up_vel_mogi(self) -> np.ndarray:
        """Up component of velocities from Mogi sources."""
        return self.vel_mogi[2::3]

    @cached_property
    def vel_tde(self) -> np.ndarray | None:
        """Velocities from TDE (triangular dislocation elements)."""
        index = self.index

        if index.tde is None:
            return None

        assert self.operators.tde is not None

        vel_tde = np.zeros(3 * self.index.n_stations)

        if index.eigen is None:
            if self.operators.tde.tde_to_velocities is None:
                raise ValueError(
                    "tde_to_velocities not available. "
                    "Rebuild operators with discard_tde_to_velocities=False."
                )
            for i, item in self.operators.tde.tde_to_velocities.items():
                # Use station_row_keep_index for rows to respect vertical flag
                tde_keep_col_index = get_keep_index_12(item.shape[1])
                vel_tde += (
                    item[self.index.station_row_keep_index, :][:, tde_keep_col_index]
                    @ self.state_vector[
                        index.tde.start_tde_col[i] : index.tde.end_tde_col[i]
                    ]
                )
            return vel_tde

        assert self.operators.eigen is not None
        for i in range(self.index.n_meshes):
            # Use station_row_keep_index to respect vertical flag
            vel_tde += (
                -self.operators.eigen.eigen_to_velocities[i][
                    self.index.station_row_keep_index, :
                ]
                @ self.state_vector[
                    index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i]
                ]
            )
        return vel_tde

    @property
    def east_vel_tde(self) -> np.ndarray | None:
        """East component of velocities from TDE."""
        if (vel := self.vel_tde) is None:
            return None
        return vel[0::3]

    @property
    def north_vel_tde(self) -> np.ndarray | None:
        """North component of velocities from TDE."""
        if (vel := self.vel_tde) is None:
            return None
        return vel[1::3]

    @property
    def up_vel_tde(self) -> np.ndarray | None:
        """Up component of velocities from TDE."""
        if (vel := self.vel_tde) is None:
            return None
        return vel[2::3]

    @cached_property
    def tde_kinematic_smooth(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to smoothed kinematic slip rate arrays."""
        return self.operators.kinematic_slip_rate(
            self.state_vector, mesh_idx=None, smooth=True
        )

    @cached_property
    def tde_kinematic(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to kinematic slip rate arrays."""
        return self.operators.kinematic_slip_rate(
            self.state_vector, mesh_idx=None, smooth=False
        )

    @property
    def tde_strike_slip_rates_kinematic(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to kinematic strike slip rate arrays."""
        return {key: val[0::2] for key, val in self.tde_kinematic.items()}

    @property
    def tde_strike_slip_rates_kinematic_smooth(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to smoothed kinematic strike slip rate arrays."""
        return {key: val[0::2] for key, val in self.tde_kinematic_smooth.items()}

    @property
    def tde_dip_slip_rates_kinematic(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to kinematic dip slip rate arrays."""
        return {key: val[1::2] for key, val in self.tde_kinematic.items()}

    @property
    def tde_dip_slip_rates_kinematic_smooth(self) -> dict[int, np.ndarray]:
        """Dictionary mapping mesh indices to smoothed kinematic dip slip rate arrays."""
        return {key: val[1::2] for key, val in self.tde_kinematic_smooth.items()}

    @property
    def tde_strike_slip_rates_coupling(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to strike slip coupling ratios."""
        kinematic = self.tde_strike_slip_rates_kinematic
        elastic = self.tde_strike_slip_rates
        if elastic is None:
            return None
        rates = {}
        for mesh_idx in kinematic:
            rates[mesh_idx] = elastic[mesh_idx] / kinematic[mesh_idx]
        return rates

    @property
    def tde_strike_slip_rates_coupling_smooth(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to smoothed strike slip coupling ratios."""
        kinematic = self.tde_strike_slip_rates_kinematic_smooth
        elastic = self.tde_strike_slip_rates
        if elastic is None:
            return None
        rates = {}
        for mesh_idx in kinematic:
            rates[mesh_idx] = elastic[mesh_idx] / kinematic[mesh_idx]
        return rates

    @property
    def tde_dip_slip_rates_coupling(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to dip slip coupling ratios."""
        kinematic = self.tde_dip_slip_rates_kinematic
        elastic = self.tde_dip_slip_rates
        if elastic is None:
            return None
        rates = {}
        for mesh_idx in kinematic:
            rates[mesh_idx] = elastic[mesh_idx] / kinematic[mesh_idx]
        return rates

    @property
    def tde_dip_slip_rates_coupling_smooth(self) -> dict[int, np.ndarray] | None:
        """Dictionary mapping mesh indices to smoothed dip slip coupling ratios."""
        kinematic = self.tde_dip_slip_rates_kinematic_smooth
        elastic = self.tde_dip_slip_rates
        if elastic is None:
            return None
        rates = {}
        for mesh_idx in kinematic:
            rates[mesh_idx] = elastic[mesh_idx] / kinematic[mesh_idx]
        return rates

    def mcmc_draw(
        self,
        draw: int | slice | list[int] | None = None,
        chain: int | slice | list[int] | None = None,
    ):
        """Get an estimation from a subset of MCMC draws/chains.

        Pass an integer, slice, or list of indices to select a subset of draws
        or chains.  ``None`` (the default) means "all draws/chains".  The
        returned :class:`Estimation` has its ``state_vector`` built from the
        mean of the selected samples and its ``mcmc_trace`` restricted to the
        selected subset.

        Args:
            draw: Draw index, slice, or list of indices to select.  ``None``
                keeps all draws.
            chain: Chain index, slice, or list of indices to select.  ``None``
                keeps all chains.
        """
        if self.mcmc_trace is None:
            raise ValueError("MCMC trace is not available.")

        from celeri.solve_mcmc import _state_vector_from_draw

        selector: dict[str, Any] = {}
        if draw is not None:
            selector["draw"] = draw
        if chain is not None:
            selector["chain"] = chain

        subsetted_trace = self.mcmc_trace.sel(**selector)

        dims_to_mean = [
            dim for dim in ("draw", "chain") if dim in subsetted_trace.posterior.dims
        ]

        posterior_point = subsetted_trace.posterior.mean(dims_to_mean)
        state_vector = _state_vector_from_draw(
            self.model, self.operators, posterior_point
        )
        return replace(self, state_vector=state_vector, mcmc_trace=subsetted_trace)

    def to_disk(self, output_dir: str | Path, *, save_operators: bool = True) -> None:
        """Save the estimation to disk.

        Args:
            output_dir: Directory to save the estimation to.
            save_operators: If True (default), save all operator arrays to disk.
                If False, only save model and index. Operators will be loaded from
                the elastic operator cache when opened.
        """
        output_dir = Path(output_dir)

        self.operators.to_disk(output_dir / "operators", save_arrays=save_operators)

        if self.mcmc_trace is not None:
            from celeri._arviz_compat import trace_to_datatree  # LEGACY-MCMC

            trace_to_datatree(self.mcmc_trace).to_zarr(
                output_dir / "mcmc_trace.zarr", consolidated=False
            )
        # We skip saving the trace, it shouldn't be needed later
        dataclass_to_disk(self, output_dir, skip={"operators", "trace", "mcmc_trace"})

    @classmethod
    def from_disk(cls, output_dir: str | Path) -> Estimation:
        """Class method to load the estimation from disk."""
        output_dir = Path(output_dir)

        operators = Operators.from_disk(output_dir / "operators")

        if (output_dir / "mcmc_trace.zarr").exists():
            import xarray as xr

            from celeri._arviz_compat import trace_from_datatree  # LEGACY-MCMC

            mcmc_trace = trace_from_datatree(
                xr.open_datatree(output_dir / "mcmc_trace.zarr", consolidated=False)
            )
        else:
            mcmc_trace = None

        extra = {"operators": operators, "mcmc_trace": mcmc_trace}
        # Skip "operator" for backwards compatibility - it was removed as a field
        # and is now a property that delegates to operators.full_dense_operator
        return dataclass_from_disk(cls, output_dir, extra=extra, skip={"operator"})


def build_estimation(
    model: Model,
    operators: Operators,
    state_vector: np.ndarray,
    *,
    state_covariance_matrix: np.ndarray | None = None,
) -> Estimation:
    """Build the estimation object.

    Args:
        model (Model): The model object.
        operators (Operators): The operators object.
        state_vector (np.ndarray): The state vector.

    Returns:
        Estimation: The estimation object.
    """
    if operators.eigen is not None:
        data_vector = _get_data_vector_eigen(model, operators.index)
        weighting_vector = _get_weighting_vector_eigen(model, operators.index)
    elif operators.tde is not None:
        data_vector = _get_data_vector(model, operators.index)
        weighting_vector = _get_weighting_vector(model, operators.index)
    else:
        data_vector = _get_data_vector_no_meshes(model, operators.index)
        weighting_vector = _get_weighting_vector_no_meshes(model, operators.index)

    return Estimation(
        data_vector=data_vector,
        weighting_vector=weighting_vector,
        state_vector=state_vector,
        operators=operators,
        state_covariance_matrix=state_covariance_matrix,
        celeri_version=celeri_version,
    )


def assemble_and_solve_dense(
    model: Model,
    *,
    eigen: bool = False,
    tde: bool = True,
    invert: bool = False,
) -> Estimation:
    operators = build_operators(model, eigen=eigen, tde=tde)

    data_vector = operators.data_vector
    weighting_vector = operators.weighting_vector
    operator = operators.full_dense_operator

    # Solve the overdetermined linear system using only a weighting vector rather than matrix
    inv_state_covariance_matrix = operator.T * weighting_vector @ operator
    if not invert:
        state_vector = linalg.solve(
            inv_state_covariance_matrix,
            operator.T * weighting_vector @ data_vector,
            assume_a="pos",
        )
    else:
        state_vector = linalg.inv(inv_state_covariance_matrix) @ (
            operator.T * weighting_vector @ data_vector
        )

    estimation = build_estimation(model, operators, state_vector)
    estimation.state_covariance_matrix = linalg.inv(inv_state_covariance_matrix)
    return estimation


def lsqlin_qp(
    C,
    d,
    reg=0,
    A=None,
    b=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    x0=None,
    opts=None,
):
    """Solve linear constrained l2-regularized least squares. Can
    handle both dense and sparse matrices. Call modeled after Matlab's
    lsqlin. It is actually wrapper around CVXOPT QP solver.

        min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
        s.t.  A * x <= b
              Aeq * x = beq
              lb <= x <= ub

    Input arguments:
        C   is m x n dense or sparse matrix
        d   is n x 1 dense matrix
        reg is regularization parameter
        A   is p x n dense or sparse matrix
        b   is p x 1 dense matrix
        Aeq is q x n dense or sparse matrix
        beq is q x 1 dense matrix
        lb  is n x 1 matrix or scalar
        ub  is n x 1 matrix or scalar

    Output arguments:
        Return dictionary, the output of CVXOPT QP.

    Dont pass matlab-like empty lists to avoid setting parameters,
    just use None:
        lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
        lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!

    Provenance notes:
    Found a few places on Github:
    - https://github.com/KasparP/PSI_simulations/blob/master/Python/SLAPMi/lsqlin.py
    - https://github.com/geospace-code/airtools/blob/main/src/airtools/lsqlin.py

    Some attribution:
    __author__ = "Valeriy Vishnevskiy", "Michael Hirsch"
    __email__ = "valera.vishnevskiy@yandex.ru"
    __version__ = "1.0"
    __date__ = "22.11.2013"
    __license__ = "MIT"
    """

    # Helper functions
    def scipy_sparse_to_spmatrix(A):
        coo = A.tocoo()
        SP = cvxopt.spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
        return SP

    def spmatrix_sparse_to_scipy(A):
        data = np.array(A.V).squeeze()
        rows = np.array(A.I).squeeze()
        cols = np.array(A.J).squeeze()
        return scipy.sparse.coo_matrix((data, (rows, cols)))

    def sparse_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return scipy.sparse.vstack([A1, A2])

    def numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        elif isinstance(A1, np.ndarray):
            return np.vstack([A1, A2])
        elif isinstance(A1, cvxopt.spmatrix):
            return np.vstack([cvxopt_to_numpy_matrix(A1).todense(), A2])

    def numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    def numpy_to_cvxopt_matrix(A):
        if A is None:
            return

        if scipy.sparse.issparse(A):
            if isinstance(A, scipy.sparse.spmatrix):
                return scipy_sparse_to_spmatrix(A)
            else:
                return A
        else:
            if isinstance(A, np.ndarray):
                if A.ndim == 1:
                    return cvxopt.matrix(A, (A.shape[0], 1), "d")
                else:
                    return cvxopt.matrix(A, A.shape, "d")
            else:
                return A

    def cvxopt_to_numpy_matrix(A):
        if A is None:
            return
        if isinstance(A, cvxopt.spmatrix):
            return spmatrix_sparse_to_scipy(A)
        elif isinstance(A, cvxopt.matrix):
            return np.asarray(A).squeeze()
        else:
            return np.asarray(A).squeeze()

    # Main function body
    if scipy.sparse.issparse(A):  # detects both np and cxopt sparse
        sparse_case = True
        # We need A to be scipy sparse, as I couldn't find how
        # CVXOPT spmatrix can be vstacked
        if isinstance(A, cvxopt.spmatrix):
            A = spmatrix_sparse_to_scipy(A)
    else:
        sparse_case = False

    C = numpy_to_cvxopt_matrix(C)
    d = numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = -d.T * C
    nvars = C.size[1]

    if reg > 0:
        if sparse_case:
            i = scipy_sparse_to_spmatrix(scipy.sparse.eye(nvars, nvars, format="coo"))
        else:
            i = cvxopt.matrix(np.eye(nvars), (nvars, nvars), "d")
        Q = Q + reg * i

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)

    if lb is not None:  # Modify 'A' and 'b' to add lb inequalities
        if lb.size == 1:
            lb = np.repeat(lb, nvars)

        if sparse_case:
            lb_A = -scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  # Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    # Convert data to CVXOPT format
    A = numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b = numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    # Set up options
    if opts is not None:
        for k, v in opts.items():
            cvxopt.solvers.options[k] = v

    # Run CVXOPT.SQP solver
    sol = cvxopt.solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol


def _build_and_solve(name: str, model: Model, *, tde: bool, eigen: bool):
    # NOTE: Used in celeri_solve.py
    logger.info(f"build_and_solve_{name}")

    # Direct solve dense linear system
    logger.info("Start: Dense assemble and solve")
    start_solve_time = timeit.default_timer()
    estimation = assemble_and_solve_dense(model, tde=True, eigen=False)
    end_solve_time = timeit.default_timer()
    logger.success(
        f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    )

    write_output(estimation)

    if model.config.plot_estimation_summary:
        from celeri.plot import plot_estimation_summary

        plot_estimation_summary(estimation)

    return estimation


def build_and_solve_dense(model: Model) -> Estimation:
    """Build and solve the dense linear system.

    Args:
        config (Config): The configuration object.
        model (Model): The model object.

    Returns:
        Estimation: The estimation object.
    """
    return _build_and_solve(
        "dense",
        model,
        tde=True,
        eigen=False,
    )


def build_and_solve_dense_no_meshes(model: Model) -> Estimation:
    """Build and solve the dense linear system without meshes.

    Args:
        config (Config): The configuration object.
        model (Model): The model object.

    Returns:
        Estimation: The estimation object.
    """
    return _build_and_solve(
        "dense_no_meshes",
        model,
        tde=False,
        eigen=False,
    )


def build_and_solve_qp_kl(model: Model) -> Estimation:
    # NOTE: Used in celeri_solve.py
    logger.info("build_and_solve_qp_kl")
    logger.info("PLACEHOLDER")

    raise NotImplementedError()
