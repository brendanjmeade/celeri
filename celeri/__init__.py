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
    get_tri_displacements,
    get_ordered_edge_nodes,
    get_mesh_edge_elements,
    block_constraints,
    slip_rate_constraints,
    get_fault_slip_rate_partials,
    get_strain_rate_centroid_operator,
    get_mogi_operator,
    get_keep_index_12,
    get_2component_index,
    get_3component_index,
    interleave2,
    post_process_estimation,
)


from .vis import (
    test_plot,
    plot_matrix_abs_log,
    plot_input_summary,
    plot_estimation_summary,
    plot_meshes,
    plot_rotation_components,
    plot_strain_rate_components_for_block,
    plot_segment_displacements,
)


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
