# Updated analysis based on actual testing
old_ok(/continuous_walk).
old_ok(/sphere_mesh).
old_ok(/collision_enabled).
old_ok(/navmesh_minimal).
old_ok(/camera_orbit).
old_ok(/input_wasd).
old_ok(/gravity_physics).
old_ok(/lod_seam_continuity).
old_ok(/no_world_cracks).
old_ok(/spherical_gravity_alignment).  # Critical missing feature
old_ok(/player_surface_orientation).   # Player orientation follows surface

must_have(F) :- old_ok(F).

# Updated new system status based on testing
new_ok(/sphere_mesh).
new_ok(/collision_enabled).
new_ok(/gravity_physics).
new_ok(/navmesh_minimal).
new_ok(/camera_orbit).
new_ok(/lod_seam_continuity).
new_ok(/no_world_cracks).
new_ok(/continuous_walk).
new_ok(/input_wasd).  # Confirmed working

# Critical missing features found in testing
missing(F) :- must_have(F), !new_ok(F).

suspect(/spherical_physics) :- missing(/spherical_gravity_alignment).
suspect(/player_controller) :- missing(/player_surface_orientation).

root_cause(/falls_off_planet) :- suspect(/spherical_physics).
root_cause(/wrong_gravity_direction) :- suspect(/spherical_physics).
root_cause(/flat_player_movement) :- suspect(/player_controller).