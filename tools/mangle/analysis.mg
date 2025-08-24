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