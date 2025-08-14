### Week 3 Low-Order Prompt (ClaudeCode)

Summary: Implement a realistic mesh asset pipeline for spherical planet worlds. Replace random primitives with grounded, themed meshes. Keep everything local-first while allowing optional cloud generation. Deliver all code, schemas, and tests end-to-end.

Objective
- Integrate a deterministic assets pipeline so Agent A emits mesh requests, a job runner generates `.glb` assets via pluggable providers, worldgen places meshes aligned to spherical surfaces, and the viewer renders meshes instead of placeholders.

Non-Negotiable Constraints
- Determinism: Every stochastic path must accept a seed and document it.
- Local-first: No required API keys. If cloud providers are supported, gate them behind env flags with clean fallbacks.
- Repo-contained: All artifacts written under `runs/` or `tests/` in this repo.
- Performance: Mesh rendering must be efficient enough for modest scenes; prefer batched draw or precomputed VBOs if trivial.
- Safety: Safe spawn zone; avoid placing meshes intersecting the player.

Deliverables
- Code and files created/edited exactly as specified below.
- Passing schema validation for manifests and basic smoke tests.
- A single command users can run to generate meshes and preview the world.

Files to Create or Edit
1) Create: `scripts/generate_meshes.py`
   - Reads `runs/assets/*_assets.json` manifests and generates `.glb` files into `runs/assets/generated/` per asset.
   - Pluggable providers with a shared interface.
   - Local provider: uses `trimesh` to procedurally synthesize a believable surrogate mesh based on tags (e.g., boulders, grass clumps, columns) when no external backend is configured.
   - Optional cloud provider (disabled by default): reads `MESH_PROVIDER=cloud` and `MESH_API_URL`, `MESH_API_KEY` env vars; submit `prompt` and download `.glb`. Must degrade gracefully if unset.
   - Writes a resolution map file `runs/assets/generated/<game>_assets_resolved.json` linking `asset_id → file_path`.

2) Create: `scripts/providers/mesh_provider_base.py`
   - `class MeshProviderBase: def generate(self, asset: dict, out_path: str) -> bool`
   - Enforce contract: returns True on success, writes `.glb` to `out_path`.

3) Create: `scripts/providers/provider_local_synth.py`
   - Implements `MeshProviderBase` using `trimesh` to procedurally synthesize meshes:
     - terrain/rock_boulder: irregular convex hull or subdivided icosphere with noise (scale from `scale_hint_meters`).
     - foliage/grass_clump: simple clump from thin planes or instanced blades merged into a mesh; keep triangle count modest.
     - structure/ruin_column/arch: parametric cylinders/arches with slight noise for erosion.
   - Deterministic by `SEED` env or provided function param; default seed stable per `asset['id']`.

4) Create: `scripts/providers/provider_cloud_api.py`
   - Implements `MeshProviderBase` using REST: POST JSON {id, prompt, category, tags} → returns a downloadable URL to `.glb`.
   - Controlled by `MESH_API_URL`, `MESH_API_KEY`. If unset/HTTP error, return False (job runner will fall back to local provider).

5) Edit: `agents/agent_a_generator.py`
   - Ensure assets manifest includes a minimum realistic set (already partially implemented). Add mapping hints to features → asset_ids so worldgen can choose assets by theme.
   - Include `assets` in `.pcc` and persist `runs/assets/<game>_assets.json`.

6) Create: `forge/tools/mesh_asset_mapper.py`
   - Utility that loads the resolved asset map and exposes: `get_asset_file(asset_id: str) -> str | None`.

7) Edit: `scripts/run_gl.py`
   - After scene generation, run meshes job: `python scripts/generate_meshes.py <manifest>` unless `--skip-meshes` is provided.
   - Export environment `MESH_PROVIDER` if passed via CLI.

8) Edit: `renderer/pcc_spherical_viewer.py`
   - Implement minimal `.glb` loading via `trimesh` → extract vertices/normals and upload to VBOs (PyOpenGL) for draw.
   - Cache loaded meshes by file path.
   - Render `{"type":"MESH","mesh_file":"...","pos":[...],"rot":[...],"scale":[...]}` properly aligned to spherical surface; keep simple immediate draw fallback if VBO not available.
   - Maintain existing continuous mouse-look and safe spawn behavior.

9) Edit: world generation path (where scene objects are produced)
   - Replace floating primitives with grounded `MESH` placements using assets from the manifest resolution map.
   - Align to surface normal; ensure no intersections with player spawn (use clearance radius).
   - Keep a small fraction of primitives for debugging, but below 10% of total non-terrain objects.

10) Create: `tests/test_assets_pipeline.py`
   - Generate a scene, build meshes with the local provider, assert that:
     - Manifest exists and validates against `schemas/assets_manifest.v1.schema.json`.
     - At least 2 `.glb` assets are generated and file sizes > 0.
     - A sample scene object is converted to `MESH` with `mesh_file` pointing to a generated `.glb`.
   - Smoke open viewer in non-interactive mode (initialize and immediately close) to ensure mesh load path doesn’t error.

11) Update: `schemas/assets_manifest.v1.schema.json`
   - Already present; confirm it is used by tests to validate manifests.

12) Create: `agents/prompting/low_order/week3_mesh_provider_README.md`
   - Brief usage: local synth provider, optional cloud provider env vars, performance tips.

Implementation Notes
- Dependencies: add to a new optional requirements file `requirements-mesh.txt` with `trimesh`, `numpy`, and any lightweight loaders used.
- Determinism: seed provider by `os.environ.get('SEED') or stable hash(asset_id)`.
- Performance: combine vertex buffers where feasible; if large, fallback to batched immediate mode.

Agent Roles and Acceptance
- Agent A: Emits assets manifest and scene that references asset ids. Deterministic seed exposed.
- Agent B: Tests (pytest) and validates schema, generation success, viewer non-crash.
- Agent C: Enforce:
  - No floating objects; align to surface.
  - Mesh coverage ≥ 80% of non-terrain objects.
  - Safe spawn: no collision at spawn; clearance radius applied.
  - Thematic coherence: assets picked based on features/material.

CLI
- Generate and run (local provider):
  - `python scripts/run_gl.py --seed 42`
  - or if manifest exists: `python scripts/generate_meshes.py runs/assets/<game>_assets.json`
- Optional cloud provider:
  - `MESH_PROVIDER=cloud MESH_API_URL=<url> MESH_API_KEY=<key> python scripts/generate_meshes.py runs/assets/<game>_assets.json`

Acceptance Checks (must pass)
- Manifests validate against schema; at least 2 assets generated.
- Scene uses `MESH` entries with valid `mesh_file` paths.
- Viewer loads at least one generated `.glb` without errors.
- Mesh coverage ≥ 80% of non-terrain objects; primitives ≤ 20%.
- Player spawn not intersecting any object.

Edits Summary (what you will implement now)
- Create providers, job runner, viewer mesh loader, mapping utility, tests, docs, and wire into `run_gl.py`.
- Maintain repo hygiene and determinism.

Output Format
- Return a concise list of edits with exact paths and brief diffs or file contents as needed. Then include the commands to run to validate.

