import pkg_resources

from .celeri import (
    read_data,
    process_station,
    process_segment,
    process_sar,
    assign_block_labels,
    merge_geodetic_data,
    get_elastic_operators,
    get_all_mesh_smoothing_matrices,
    get_all_mesh_smoothing_matrices_simple,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
