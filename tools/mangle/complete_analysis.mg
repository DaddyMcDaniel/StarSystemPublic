# Old system guarantees
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

# New system status
new_ok(/sphere_mesh).
new_ok(/collision_enabled).
new_ok(/gravity_physics).
new_ok(/navmesh_minimal).
new_ok(/camera_orbit).
new_ok(/lod_seam_continuity).
new_ok(/no_world_cracks).
new_ok(/continuous_walk).

# Analysis rules
missing(F) :- must_have(F), !new_ok(F).

suspect(/navmesh_pipeline) :- missing(/navmesh_minimal).
suspect(/physics_or_collision) :- missing(/collision_enabled).
suspect(/physics_or_collision) :- missing(/gravity_physics).
suspect(/lod_or_streaming) :- missing(/lod_seam_continuity).
suspect(/lod_or_streaming) :- missing(/no_world_cracks).
suspect(/controls_or_camera) :- missing(/camera_orbit).
suspect(/controls_or_camera) :- missing(/input_wasd).
suspect(/worldgen_mesh) :- missing(/sphere_mesh).

root_cause(/walkability_broken) :- suspect(/navmesh_pipeline).
root_cause(/fall_through_world) :- suspect(/physics_or_collision).
root_cause(/seams_or_cracks) :- suspect(/lod_or_streaming).
root_cause(/camera_or_input_regression) :- suspect(/controls_or_camera).
root_cause(/mesh_pipeline_regression) :- suspect(/worldgen_mesh).