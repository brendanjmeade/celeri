import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import path
from scipy.spatial.distance import cdist

from celeri import celeri_closure
from celeri.celeri_util import sph2cart
from celeri.config import Config, get_config
from celeri.constants import GEOID, RADIUS_EARTH
from celeri.mesh import Mesh


@dataclass
class Model:
    """Represents a problem configuration for Celeri fault slip rate modeling.

    Stores indices, meshes, operators, and various data components needed
    for solving interseismic coupling and fault slip rate problems.

    Attributes:
        meshes: list[Mesh]
        segment: pd.DataFrame
        block: pd.DataFrame
        station: pd.DataFrame
        mogi: pd.DataFrame
        config: Config
        closure: BlockClosureResult
        sar: pd.DataFrame
    """

    meshes: list[Mesh]
    segment: pd.DataFrame
    block: pd.DataFrame
    station: pd.DataFrame
    mogi: pd.DataFrame
    config: Config
    closure: celeri_closure.BlockClosureResult
    sar: pd.DataFrame

    @property
    def segment_mesh_indices(self):
        n_segment_meshes = np.max(self.segment.mesh_file_index).astype(int) + 1
        return list(range(n_segment_meshes))

    @property
    def total_mesh_points(self):
        return sum([self.meshes[idx].n_tde for idx in self.segment_mesh_indices])

    @classmethod
    def from_config(cls, config: Config) -> "Model":
        """Create a Model instance from a Config object."""
        return build_model(config)

    def to_disk(self, output_path: str | Path | None = None):
        """Save the model to disk."""
        if output_path is None:
            output_path = self.config.output_path
        if output_path is None:
            raise ValueError("Output path must be specified.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.segment.to_parquet(output_path / "segment.parquet")
        self.block.to_parquet(output_path / "block.parquet")
        self.station.to_parquet(output_path / "station.parquet")
        self.mogi.to_parquet(output_path / "mogi.parquet")
        self.sar.to_parquet(output_path / "sar.parquet")
        with (output_path / "config.json").open("w") as f:
            f.write(self.config.model_dump_json())
        for i, mesh in enumerate(self.meshes):
            mesh.to_disk(output_path / f"meshes/{i:05d}/")
        logger.success(f"Model saved to {output_path}")

    @classmethod
    def from_disk(cls, path: str | Path) -> "Model":
        """Load a Model instance from disk."""
        path = Path(path)
        # read the config using pydantic:
        config = Config.model_validate_json(
            (path / "config.json").read_text(encoding="utf-8")
        )
        segment = pd.read_parquet(path / "segment.parquet")
        block = pd.read_parquet(path / "block.parquet")
        station = pd.read_parquet(path / "station.parquet")
        mogi = pd.read_parquet(path / "mogi.parquet")
        sar = pd.read_parquet(path / "sar.parquet")

        meshes = [
            Mesh.from_disk(mesh_file) for mesh_file in sorted(path.glob("meshes/*"))
        ]

        closure = celeri_closure.BlockClosureResult.from_segments(segment)

        return cls(
            meshes=meshes,
            segment=segment,
            block=block,
            station=station,
            mogi=mogi,
            config=config,
            closure=closure,
            sar=sar,
        )


def read_data(config: Config):
    logger.info("Reading data files")
    # Read segment data
    segment = pd.read_csv(config.segment_file_name)
    segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]

    for name in ["ss", "ds", "ts"]:
        if f"{name}_rate_bound_flag" not in segment.columns:
            segment[f"{name}_rate_bound_flag"] = 0.0
        if f"{name}_rate_bound_min" not in segment.columns:
            segment[f"{name}_rate_bound_min"] = -1.0
        if f"{name}_rate_bound_max" not in segment.columns:
            segment[f"{name}_rate_bound_max"] = 1.0

        if f"{name}_reg_flag" not in segment.columns:
            segment[f"{name}_reg_flag"] = 1
            segment.loc[segment[f"{name}_rate_bound_flag"] == 1, f"{name}_reg_flag"] = 0

    logger.success(f"Read: {config.segment_file_name}")

    # Read block data
    block = pd.read_csv(config.block_file_name)
    block = block.loc[:, ~block.columns.str.match("Unnamed")]
    logger.success(f"Read: {config.block_file_name}")

    # Read mesh data - List of dictionary version
    meshes = []
    meshes = [Mesh.from_params(mesh_param) for mesh_param in config.mesh_params]

    station = pd.read_csv(config.station_file_name)
    station = station.loc[:, ~station.columns.str.match("Unnamed")]
    logger.success(f"Read: {config.station_file_name}")

    # Read Mogi source data
    if config.mogi_file_name is None:
        columns = pd.Index(
            [
                "name",
                "lon",
                "lat",
                "depth",
                "volume_change_flag",
                "volume_change",
                "volume_change_sig",
            ]
        )
        mogi = pd.DataFrame(columns=columns)
        logger.info("No mogi_file_name")
    else:
        mogi = pd.read_csv(config.mogi_file_name)
        mogi = mogi.loc[:, ~mogi.columns.str.match("Unnamed")]
        logger.success(f"Read: {config.mogi_file_name}")

    # Read SAR data
    if config.sar_file_name is None:
        columns = pd.Index(
            [
                "lon",
                "lat",
                "depth",
                "line_of_sight_change_val",
                "line_of_sight_change_sig",
                "look_vector_x",
                "look_vector_y",
                "look_vector_z",
                "reference_point_x",
                "reference_point_y",
            ]
        )
        sar = pd.DataFrame(columns=columns)
        logger.info("No sar_file_name")
    else:
        sar = pd.read_csv(config.sar_file_name)
        sar = sar.loc[:, ~sar.columns.str.match("Unnamed")]
        logger.success(f"Read: {config.sar_file_name}")
    return segment, block, meshes, station, mogi, sar


def create_output_folder(config: Config):
    config.output_path.mkdir(parents=True, exist_ok=True)


def build_model(
    config_path: str | Path | Config,
    *,
    override_segment: pd.DataFrame | None = None,
    override_block: pd.DataFrame | None = None,
    override_meshes: list[Mesh] | None = None,
    override_station: pd.DataFrame | None = None,
    override_mogi: pd.DataFrame | None = None,
    override_sar: pd.DataFrame | None = None,
) -> Model:
    if isinstance(config_path, Config):
        config = config_path
    else:
        config = get_config(config_path)
    create_output_folder(config)
    segment, block, meshes, station, mogi, sar = read_data(config)

    if override_segment is not None:
        segment = override_segment
    if override_block is not None:
        block = override_block
    if override_meshes is not None:
        meshes = override_meshes
    if override_station is not None:
        station = override_station
    if override_mogi is not None:
        mogi = override_mogi
    if override_sar is not None:
        sar = override_sar

    station = process_station(station, config)
    segment = process_segment(segment, config, meshes)
    sar = process_sar(sar, config)
    closure, segment, station, block, mogi, sar = assign_block_labels(
        segment, station, block, mogi, sar
    )

    return Model(
        meshes=meshes,
        segment=segment,
        block=block,
        station=station,
        config=config,
        mogi=mogi,
        sar=sar,
        closure=closure,
    )


def process_station(station, config):
    if bool(config.unit_sigmas):  # Assign unit uncertainties, if requested
        station.east_sig = np.ones_like(station.east_sig)
        station.north_sig = np.ones_like(station.north_sig)
        station.up_sig = np.ones_like(station.up_sig)

    station["depth"] = np.zeros_like(station.lon)
    station["x"], station["y"], station["z"] = sph2cart(
        station.lon, station.lat, RADIUS_EARTH
    )
    station = station.drop(np.where(station.flag == 0)[0])
    station = station.reset_index(drop=True)
    return station


def process_sar(sar, config):
    """Preprocessing of SAR data."""
    if sar.empty:
        sar["depth"] = np.zeros_like(sar.lon)
        sar["x"], sar["y"], sar["z"] = sph2cart(sar.lon, sar.lat, RADIUS_EARTH)
        sar["block_label"] = -1 * np.ones_like(sar.x)
    else:
        sar["dep"] = []
        sar["x"] = []
        sar["y"] = []
        sar["x"] = []
        sar["block_label"] = []
    return sar


def process_segment(segment, config, meshes):
    """Add derived fields to segment dataframe."""
    segment["x1"], segment["y1"], segment["z1"] = sph2cart(
        segment.lon1, segment.lat1, RADIUS_EARTH
    )
    segment["x2"], segment["y2"], segment["z2"] = sph2cart(
        segment.lon2, segment.lat2, RADIUS_EARTH
    )

    segment = order_endpoints_sphere(segment)

    segment["length"] = np.zeros(len(segment))
    segment["azimuth"] = np.zeros(len(segment))
    for i in range(len(segment)):
        segment.azimuth.values[i], _, segment.length.values[i] = GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )  # Segment azimuth, Segment length in meters

    # This calculation needs to account for the periodic nature of longitude.
    # Calculate the periodic longitudinal separation.
    # @BJM: Is this better done with GEIOD?
    sep = segment.lon2 - segment.lon1
    periodic_lon_separation = np.where(
        sep > 180, sep - 360, np.where(sep < -180, sep + 360, sep)
    )
    segment["mid_lon_plate_carree"] = (
        segment.lon1.values + periodic_lon_separation / 2.0
    )

    # No worries for latitude because there's no periodicity.
    segment["mid_lat_plate_carree"] = (segment.lat1.values + segment.lat2.values) / 2.0
    segment["mid_lon"] = np.zeros_like(segment.lon1)
    segment["mid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment.mid_lon.values[i], segment.mid_lat.values[i] = GEOID.npts(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i], 1
        )[0]
    segment.mid_lon.values[segment.mid_lon < 0.0] += 360.0

    segment["mid_x"], segment["mid_y"], segment["mid_z"] = sph2cart(
        segment.mid_lon, segment.mid_lat, RADIUS_EARTH
    )
    segment = locking_depth_manager(segment, config)
    segment = zero_mesh_segment_locking_depth(segment, meshes)
    segment = segment_centroids(segment)
    return segment


def order_endpoints_sphere(segment):
    """Endpoint ordering function, placing west point first.
    This converts the endpoint coordinates from spherical to Cartesian,
    then takes the cross product to test for ordering (i.e., a positive z
    component of cross(point1, point2) means that point1 is the western
    point). This method works for both (-180, 180) and (0, 360) longitude
    conventions.
    BJM: Not sure why cross product approach was definitely not working in
    python so I revereted to relative longitude check which sould be fine because
    we're always in 0-360 space.
    """
    segment_copy = copy.deepcopy(segment)
    endpoints1 = np.transpose(np.array([segment.x1, segment.y1, segment.z1]))
    endpoints2 = np.transpose(np.array([segment.x2, segment.y2, segment.z2]))
    cross_product = np.cross(endpoints1, endpoints2)
    swap_endpoint_idx = np.where(cross_product[:, 2] < 0)
    segment_copy.lon1.values[swap_endpoint_idx] = segment.lon2.values[swap_endpoint_idx]
    segment_copy.lat1.values[swap_endpoint_idx] = segment.lat2.values[swap_endpoint_idx]
    segment_copy.lon2.values[swap_endpoint_idx] = segment.lon1.values[swap_endpoint_idx]
    segment_copy.lat2.values[swap_endpoint_idx] = segment.lat1.values[swap_endpoint_idx]
    return segment_copy


def locking_depth_manager(segment, config):
    """This function assigns the locking depths given in the config file to any
    segment that has the same locking depth flag.  Segments with flag =
    0, 1 are untouched.
    """
    segment = segment.copy(deep=True)
    segment.locking_depth.values[segment.locking_depth_flag == 2] = (
        config.locking_depth_flag2
    )
    segment.locking_depth.values[segment.locking_depth_flag == 3] = (
        config.locking_depth_flag3
    )
    segment.locking_depth.values[segment.locking_depth_flag == 4] = (
        config.locking_depth_flag4
    )
    segment.locking_depth.values[segment.locking_depth_flag == 5] = (
        config.locking_depth_flag5
    )

    if bool(config.locking_depth_override_flag):
        segment.locking_depth.values = config.locking_depth_override_value
    return segment


def zero_mesh_segment_locking_depth(segment, meshes):
    """This function sets the locking depths of any segments that trace
    a mesh to zero, so that they have no rectangular elastic strain
    contribution, as the elastic strain is accounted for by the mesh.

    To have its locking depth set to zero, the segment's mesh_flag
    and mesh_file_index fields must not be equal to zero but also
    less than the number of available mesh files.
    """
    segment = segment.copy(deep=True)
    toggle_off = np.where(
        (segment.mesh_flag != 0)
        & (segment.mesh_file_index >= 0)
        & (segment.mesh_file_index <= len(meshes))
    )[0]
    segment.locking_depth.values[toggle_off] = 0
    return segment


def segment_centroids(segment):
    """Calculate segment centroids."""
    segment["centroid_x"] = np.zeros_like(segment.lon1)
    segment["centroid_y"] = np.zeros_like(segment.lon1)
    segment["centroid_z"] = np.zeros_like(segment.lon1)
    segment["centroid_lon"] = np.zeros_like(segment.lon1)
    segment["centroid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment_forward_azimuth, _, _ = GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )
        segment_down_dip_azimuth = segment_forward_azimuth + 90.0 * np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        )
        azimuth_xy_cartesian = (segment.y2[i] - segment.y1[i]) / (
            segment.x2[i] - segment.x1[i]
        )
        azimuth_xy_cartesian = np.arctan(-1.0 / azimuth_xy_cartesian)
        segment.centroid_z.values[i] = segment.locking_depth[i] / 2.0
        segment_down_dip_distance = segment.centroid_z[i] / np.abs(
            np.tan(np.deg2rad(segment.dip[i]))
        )
        (
            segment.centroid_lon.values[i],
            segment.centroid_lat.values[i],
            _,
        ) = GEOID.fwd(
            segment.mid_lon[i],
            segment.mid_lat[i],
            segment_down_dip_azimuth,
            segment_down_dip_distance,
        )
        segment.centroid_x.values[i] = segment.mid_x[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.cos(azimuth_xy_cartesian)
        segment.centroid_y.values[i] = segment.mid_y[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.sin(azimuth_xy_cartesian)
    segment.centroid_lon.values[segment.centroid_lon < 0.0] += 360.0
    return segment


def inpolygon(xq, yq, xv, yv):
    """Check if points (xq, yq) are inside polygon (xv, yv) using matplotlib."""
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def assign_block_labels(segment, station, block, mogi, sar):
    """Ben Thompson's implementation of the half edge approach to the
    block labeling problem and east/west assignment.
    """
    # segment = split_segments_crossing_meridian(segment)
    segment = segment.copy(deep=True)
    station = station.copy(deep=True)
    block = block.copy(deep=True)
    mogi = mogi.copy(deep=True)
    sar = sar.copy(deep=True)

    closure = celeri_closure.BlockClosureResult.from_segments(segment)
    labels = celeri_closure.get_segment_labels(closure)

    segment["west_labels"] = labels[:, 0]
    segment["east_labels"] = labels[:, 1]

    # Check for unprocessed indices
    unprocessed_indices = np.union1d(
        np.where(segment["east_labels"] < 0),
        np.where(segment["west_labels"] < 0),
    )
    if len(unprocessed_indices) > 0:
        logger.warning("Found unproccessed indices")

    # Find relative areas of each block to identify an external block
    block["area_steradians"] = -1 * np.ones(len(block))
    block["area_plate_carree"] = -1 * np.ones(len(block))

    # Debug: Check if number of blocks matches
    if closure.n_polygons() != len(block):
        logger.error(
            f"Mismatch: {closure.n_polygons()} polygons but {len(block)} blocks in input file"
        )
    else:
        logger.info(
            f"Closed blocks: {closure.n_polygons()}, equal to block file blocks: {len(block)}"
        )

    # Also test with matplotlib path for comparison
    logger.info("Testing for existence of interior points")
    for i in range(closure.n_polygons()):
        vs = closure.polygons[i].vertices
        contained = inpolygon(
            block.interior_lon.to_numpy(),
            block.interior_lat.to_numpy(),
            vs[:, 0],
            vs[:, 1],
        )

        # Case 1: No interior points
        if np.sum(contained) == 0:
            logger.warning(f"Polygon {i = } has no interior point")
            logger.warning(f"{block.interior_lon.to_numpy() = }")
            logger.warning(f"{block.interior_lat.to_numpy() = }")
            logger.warning(f"{vs[:, 0] = }")
            logger.warning(f"{vs[:, 1] = }")

            # Debug figure
            plt.figure()
            plt.title(f"polygon {i = }")
            for j in range(len(segment)):
                plt.plot(
                    [segment.lon1[j], segment.lon2[j]],
                    [segment.lat1[j], segment.lat2[j]],
                    "-k",
                )
            plt.plot(vs[:, 0], vs[:, 1], "-r")
            plt.plot(block.interior_lon.to_numpy(), block.interior_lat.to_numpy(), ".b")
            lon_min, lon_max = vs[:, 0].min(), vs[:, 0].max()
            lat_min, lat_max = vs[:, 1].min(), vs[:, 1].max()
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            padding = max(lon_range, lat_range) * 0.1
            plt.xlim(lon_min - padding, lon_max + padding)
            plt.ylim(lat_min - padding, lat_max + padding)
            plt.show()

        # Case 2: One interior point.  Nothing to do

        # Case 3: More than one interior point.
        # The first condition checks for more than one interior point
        # The second conditional is to ensure that we're not pickup up all the interior
        # points intside the exterior block
        if np.sum(contained) > 1 and np.sum(contained) < np.round(len(block) / 2):
            logger.warning(
                f"Polygon {i = } has has {np.sum(contained)} interior points"
            )
            logger.warning(f"{block.interior_lon.to_numpy() = }")
            logger.warning(f"{block.interior_lat.to_numpy() = }")
            logger.warning(f"{vs[:, 0] = }")
            logger.warning(f"{vs[:, 1] = }")

            # Debug figure
            plt.figure()
            plt.title(f"polygon {i = }")
            for j in range(len(segment)):
                plt.plot(
                    [segment.lon1[j], segment.lon2[j]],
                    [segment.lat1[j], segment.lat2[j]],
                    "-k",
                )
            for j in range(len(vs[:, 0])):
                plt.text(vs[j, 0], vs[j, 1], f"{j}", fontsize=6)
            plt.plot(vs[:, 0], vs[:, 1], "-r")
            plt.plot(block.interior_lon.to_numpy(), block.interior_lat.to_numpy(), ".b")
            lon_min, lon_max = vs[:, 0].min(), vs[:, 0].max()
            lat_min, lat_max = vs[:, 1].min(), vs[:, 1].max()
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            padding = max(lon_range, lat_range) * 0.1
            plt.xlim(lon_min - padding, lon_max + padding)
            plt.ylim(lat_min - padding, lat_max + padding)
            plt.show()

    # Assign block labels points to block interior points
    block["block_label"] = closure.assign_points(
        block.interior_lon.to_numpy(), block.interior_lat.to_numpy()
    )

    # I copied this from the bottom of:
    # https://stackoverflow.com/questions/39992502/rearrange-rows-of-pandas-dataframe-based-on-list-and-keeping-the-order
    # and I definitely don't understand it all but emperically it seems to work.
    block = (
        block.set_index(block.block_label, append=True)
        .sort_index(level=1)
        .reset_index(1, drop=True)
    )
    block = block.reset_index()
    block = block.loc[:, ~block.columns.str.match("index")]

    # Assign block labels to GPS stations
    if not station.empty:
        station["block_label"] = closure.assign_points(
            station.lon.to_numpy(), station.lat.to_numpy()
        )

    # Assign block labels to SAR locations
    if not sar.empty:
        sar["block_label"] = closure.assign_points(
            sar.lon.to_numpy(), sar.lat.to_numpy()
        )

    # Assign block labels to Mogi sources
    if not mogi.empty:
        mogi["block_label"] = closure.assign_points(
            mogi.lon.to_numpy(), mogi.lat.to_numpy()
        )

    return closure, segment, station, block, mogi, sar


def assign_mesh_segment_labels(
    model: Model, mesh_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a mesh, finds the closest segment midpoint and assigns
    the east and west block labels from that segment.

    Returns:
        closest_segment_idx: indices of closest segments
        east_labels: block labels on eastern side
        west_labels: block labels on western side
    """
    # Find subset of segments that are replaced by this mesh
    seg_replace_idx = np.where(
        (model.segment.mesh_flag != 0) & (model.segment.mesh_file_index == mesh_idx)
    )

    # Find closest segment midpoint to each element centroid, using scipy.spatial.cdist
    model.meshes[mesh_idx].closest_segment_idx = seg_replace_idx[0][
        cdist(
            np.array(
                [
                    model.meshes[mesh_idx].lon_centroid,
                    model.meshes[mesh_idx].lat_centroid,
                ]
            ).T,
            np.array(
                [
                    model.segment.mid_lon[seg_replace_idx[0]],
                    model.segment.mid_lat[seg_replace_idx[0]],
                ]
            ).T,
        ).argmin(axis=1)
    ]

    # Add segment labels to elements
    closest_segment_idx = model.meshes[mesh_idx].closest_segment_idx
    assert closest_segment_idx is not None

    east_labels = np.array(model.segment.east_labels[closest_segment_idx])
    west_labels = np.array(model.segment.west_labels[closest_segment_idx])

    assert east_labels is not None and west_labels is not None
    return east_labels, west_labels, closest_segment_idx
