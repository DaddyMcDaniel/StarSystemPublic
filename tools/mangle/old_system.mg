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