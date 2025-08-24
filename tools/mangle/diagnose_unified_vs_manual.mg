# Baseline: old manual loop guarantees
old_ok(/continuous_walk).
old_ok(/sphere_mesh).
old_ok(/collision_enabled).
old_ok(/navmesh_minimal).
old_ok(/camera_orbit).
old_ok(/input_wasd).
old_ok(/gravity_physics).
old_ok(/lod_seam_continuity).
old_ok(/no_world_cracks).

must_have(F) :- old_ok(F).

# Load new system facts produced by the scanner below
::load new_system.mg

# Diff
missing(F) :- must_have(F), not new_ok(F).

# Heuristic suspects -> coarse root causes
suspect(/navmesh_pipeline) :- missing(/navmesh_minimal).
suspect(/physics_or_collision) :- missing(/collision_enabled); missing(/gravity_physics).
suspect(/lod_or_streaming) :- missing(/lod_seam_continuity); missing(/no_world_cracks).
suspect(/controls_or_camera) :- missing(/camera_orbit); missing(/input_wasd).
suspect(/worldgen_mesh) :- missing(/sphere_mesh).

root_cause(/walkability_broken) :- suspect(/navmesh_pipeline).
root_cause(/fall_through_world) :- suspect(/physics_or_collision).
root_cause(/seams_or_cracks) :- suspect(/lod_or_streaming).
root_cause(/camera_or_input_regression) :- suspect(/controls_or_camera).
root_cause(/mesh_pipeline_regression) :- suspect(/worldgen_mesh).

# Report
::print new_ok.
::print missing.
::print suspect.
::print root_cause.
