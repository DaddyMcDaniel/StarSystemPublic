## Mini-Planet Core Components and Interactions

### Purpose
Concise map of the systems required to generate, render, and navigate a mini-planet, and how they interact at runtime and during baking.

### Components
- **PCC Language & Schema**
  - Files: `agents/agent_d/schema/pcc_schema_v1.json`, `agents/agent_d/schema/pcc_terrain_nodes.py`
  - Role: Declarative definition of terrain graphs, seeded determinism, and generation parameters.

- **Terrain Generation**
  - Files: `agents/agent_d/terrain/heightfield.py`, `agents/agent_d/terrain/noise_nodes.py`
  - Role: Computes scalar height/displacement from procedural graphs, supports FBM/ridged/domain-warped stacks.

- **SDF + Marching Cubes (optional subsurface/caves)**
  - Files: `agents/agent_d/sdf/sdf_evaluator.py`, `agents/agent_d/marching_cubes/marching_cubes.py`, `agents/agent_d/marching_cubes/chunk_cave_generator.py`
  - Role: Volumetric feature authoring and extraction; merges with surface terrain.

- **Planet Mesh Base**
  - Files: `agents/agent_d/mesh/cubesphere.py`, `agents/agent_d/mesh/shading_basis.py`
  - Role: Spherical parameterization and initial vertex layout; tangent basis for normal mapping.

- **Chunking & Runtime LOD**
  - Files: `agents/agent_d/mesh/quadtree_chunking.py`, `agents/agent_d/mesh/runtime_lod.py`, `agents/agent_d/mesh/crack_prevention.py`
  - Role: Spatial subdivision, view-dependent detail, and seam/crack prevention across LOD boundaries.

- **Baking, Determinism, and Performance**
  - Files: `agents/agent_d/baking/deterministic_baking.py`, `agents/agent_d/determinism/seed_threading.py`, `agents/agent_d/performance/async_streaming.py`, `agents/agent_d/performance/buffer_pools.py`
  - Role: Reproducible outputs, precomputation, streaming, and memory reuse.

- **Rendering & Lighting**
  - Files: `renderer/pcc_game_viewer.py`, `renderer/pcc_spherical_viewer.py`, `agents/agent_d/lighting/lighting_system.py`, `agents/agent_d/lighting/shadow_mapping.py`, `agents/agent_d/lighting/ssao.py`, `agents/agent_d/materials/mikktspace_tangents.py`
  - Role: Realtime visualization with shadows, SSAO, normal mapping, and viewer controls.

- **Navigation & Debugging**
  - Files: `agents/agent_d/camera_tools/debug_camera.py`, `agents/agent_d/hud/lod_statistics_hud.py`, `agents/agent_d/debug_ui/debug_toggles.py`
  - Role: Movement, diagnostics, and developer UX.

- **Validation**
  - Files: `agents/agent_d/validation/seam_consistency_validator.py`, `agents/agent_d/validation/terrain_quality_validator.py`
  - Role: Automated checks for seams, topology, and quality gates.

### Data Flow (High-level)
1. PCC config -> Terrain node graph instantiation (seeded).
2. Base mesh (cubesphere) -> Vertex displacement via heightfield.
3. Optional SDF volume -> Marching cubes -> Merge surface + caves.
4. Quadtree chunking -> Crack prevention metadata -> LOD hierarchy.
5. Deterministic baking -> Serialized chunk assets + metadata.
6. Viewer load -> Async streaming + LOD -> Lighting/material pass.
7. Camera/navigation -> Continuous LOD updates -> HUD/diagnostics.

### Runtime Interaction Sequence (Simplified)
1. Initialize renderer and lighting; load baked planet manifest.
2. Spawn camera; register LOD manager, streaming, and culling.
3. Each frame:
   - Update camera state and visibility sets.
   - Stream in/out chunks; select LOD per tile.
   - Apply crack prevention on neighbors when LOD changes.
   - Submit draw calls with material/lighting pipelines.
   - Present HUD and optional debug overlays.

### Key Contracts
- Heightfield interface: sample(height) given spherical direction or parametric UV.
- Chunk interface: provides vertex/index buffers, bounds, neighbors, LOD level.
- Baking manifest: lists chunk URIs, seam topology, and material flags.
- Viewer API: accepts baked manifest and rendering options; exposes camera control.

### Validation & QA Hooks
- Seam validator executes per LOD transition set and face seams.
- Determinism tests run generation twice per seed -> byte-compare manifests.
- Performance probes around streaming and buffer pool hot paths.

### Integration Notes (Godot Fork)
- Export baked chunks as binary + JSON manifest; ensure index/vertex compression flags.
- Mirror LOD selection and seam rules in Godot nodes; reuse seed/config.
- Map material parameters to Godot shaders (normal/roughness/ao).


