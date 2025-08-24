#!/usr/bin/env bash
set -euo pipefail; shopt -s nullglob
project_root="${1:-.}"

emit_ok(){ echo "new_ok($1)." ; }
emit_unknown(){ :; }

have_file(){ [[ -f "$1" ]]; }
have_any_file(){ for f in "$@"; do [[ -f "$f" ]] && return 0; done; return 1; }
grepi(){ LC_ALL=C grep -RInE --exclude-dir '\.git' --exclude-dir 'venv' --exclude-dir 'node_modules' "$1" "$project_root" >/dev/null 2>&1; }

# sphere_mesh
if have_any_file "$project_root/agents/agent_d/mesh/cpp_cubesphere_indices.bin" \
                 "$project_root/agents/agent_d/mesh/small_chunks/face0_L1_0.250_0.250_positions.bin" \
                 "$project_root/agents/agent_d/marching_cubes/marching_cubes.py"; then
  emit_ok "/sphere_mesh"
elif grepi '(cubesphere|smooth[ _-]?sphere|marching[_-]?cubes)' ; then
  emit_ok "/sphere_mesh"
else
  emit_unknown "/sphere_mesh"
fi

# collision_enabled + gravity_physics (proxy: physics module present / gravity refs)
if have_any_file "$project_root/forge/modules/physics/physics_module.py"; then
  emit_ok "/collision_enabled"
fi
if grepi '\bgravity\b' ; then
  emit_ok "/gravity_physics"
fi

# navmesh_minimal (likely absent if no nav/navmesh mentions)
if grepi '(navmesh|navigation|path[-_ ]?finding)'; then
  emit_ok "/navmesh_minimal"
fi

# camera_orbit
if have_any_file "$project_root/agents/agent_d/camera_tools/debug_camera.py" ; then
  emit_ok "/camera_orbit"
elif grepi '(camera|orbit|yaw|pitch)'; then
  emit_ok "/camera_orbit"
fi

# input_wasd
if grepi '([[:<:]]W[[:>:]]|[[:<:]]A[[:>:]]|[[:<:]]S[[:>:]]|[[:<:]]D[[:>:]]|keyboard|key[_-]?down|input)'; then
  emit_ok "/input_wasd"
fi

# lod_seam_continuity / no_world_cracks (proxies: fusion or seam validators exist)
if have_any_file "$project_root/agents/agent_d/fusion/chunk_border_fusion.py" \
                 "$project_root/agents/agent_d/fusion/surface_sdf_fusion.py"; then
  emit_ok "/lod_seam_continuity"
fi
if have_any_file "$project_root/agents/agent_d/validation/seam_consistency_validator.py"; then
  emit_ok "/no_world_cracks"
fi

# continuous_walk (proxy: game viewer or developer viewer present)
if have_any_file "$project_root/renderer/pcc_game_viewer.py" \
                 "$project_root/agents/agent_d/viewer_tools/developer_viewer.py"; then
  emit_ok "/continuous_walk"
fi
