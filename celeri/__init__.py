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
    get_block_rotation_operator,
    get_global_float_block_rotation_operator,
    block_constraints,
    slip_rate_constraints,
    get_fault_slip_rate_partials,
    get_strain_rate_centroid_operator,
    get_mogi_operator,
    post_process_estimation,
    plot_input_summary,
    plot_estimation_summary,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
