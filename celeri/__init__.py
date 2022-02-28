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
    get_rotation_to_velocities_partials,
    get_global_float_block_rotation_partials,
    get_tri_displacements,
    get_ordered_edge_nodes,
    get_mesh_edge_elements,
    get_block_motion_constraints,
    get_slip_rate_constraints,
    get_rotation_to_slip_rate_partials,
    get_block_strain_rate_to_velocities_partials,
    get_mogi_to_velocities_partials,
    get_tde_slip_rate_constraints,
    get_keep_index_12,
    get_2component_index,
    get_3component_index,
    interleave2,
    post_process_estimation,
    get_tde_to_velocities,
    get_shared_sides,
    write_output,
    create_output_folder,
    assemble_and_solve_dense,
    RUN_NAME,
)


from .celeri_vis import (
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
