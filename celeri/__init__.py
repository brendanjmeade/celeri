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
    get_index,
    get_data_vector,
    get_weighting_vector,
    get_full_dense_operator_block_only,
    get_weighting_vector_single_mesh_for_col_norms,
    get_tde_to_velocities_single_mesh,
    get_elastic_operator_single_mesh,
    get_segment_station_operator_okada,
    get_elastic_operators_okada,
    get_command,
    get_logger,
    matvec,
    matvec_wrapper,
    rmatvec,
    rmatvec_wrapper,
    post_process_estimation_hmatrix,
    get_h_matrices_for_tde_meshes,
    align_velocities,
    process_args,
    get_processed_data_structures,
    plot_input_summary,
    plot_estimation_summary,
    build_and_solve_hmatrix,
)


from .celeri_vis import (
    plot_matrix_abs_log,
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
