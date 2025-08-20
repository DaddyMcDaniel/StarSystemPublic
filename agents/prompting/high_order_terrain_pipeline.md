Summary: Terrain Pipeline High-Order Prompt (governor). Cube-sphere planet, correct tangents, heightfield + SDF fusion, static LOD, determinism, and performance/diagnostics.

```xml
<high-order-prompt version="5.0"
 model="claude sonnet 4, unless specified otherwise in Low-order prompt"
 role="governor">
  <!-- Self-contained terrain pipeline HOD. Do not rely on external text. -->
  <project id="starsystem" name="StarSystem + The Forge">
    <phase week="5" name="Terrain Pipeline: Cube-sphere → Heightfield/SDF → LOD/Diagnostics" seed="${GLOBAL_SEED}" timezone="UTC"/>
    <vision>
      <north_star>Deterministic, reproducible planet terrain pipeline with correct shading and seams, scalable LOD, and clear diagnostics.</north_star>
      <visual_excellence>Smoothly shaded geometry with correct normals/tangents, crack-free chunk boundaries, and coherent materials.</visual_excellence>
      <operation_robustness>Runs with or without OpenGL; clear fallbacks and harnesses ensure reproducibility and debuggability.</operation_robustness>
    </vision>

    <goals>
      <primary>
        <goal>Implement a cube-sphere primitive with heightfield displacement and optional SDF fusion, chunked with per-face quadtrees.</goal>
        <goal>Ensure tangent-space correctness (MikkTSpace-compatible), and prevent cracks between LOD levels via stitching/skirts.</goal>
        <goal>Provide runtime LOD + frustum culling and diagnostics HUD/toggles for verification.</goal>
      </primary>
      <secondary>
        <goal>Author an example “hero” planet using the pipeline.</goal>
        <goal>Harden PCC schemas and determinism harness for the terrain pipeline.</goal>
        <goal>Performance pass with multithreaded bake and streaming.</goal>
      </secondary>
    </goals>
  </project>

  <non-negotiables>
    <interface>MCP-only (official Python SDK); tools carry explicit input/output $id schemas.</interface>
    <cloud-llms>forbidden</cloud-llms>
    <secrets>do-not-read-or-require</secrets>
    <determinism rng="pcg64">
      <seed-cascade>global_seed → world_seed → tool_seed → chip_seed</seed-cascade>
      <float-tolerance>1e-9</float-tolerance>
      <time-source>UTC only; no wallclock in logic</time-source>
    </determinism>
    <artifacts dir="./runs/${UTC_ISO}--${GLOBAL_SEED}" latest_symlink="./runs/latest" manifest="run_manifest.json"/>
    <installs approval_flag="APPROVE_INSTALL" behavior="print-plan-and-exit-if-unset"/>
    <versions python="3.10..3.12" opengl="optional with graceful fallbacks"/>
    <operation_modes>
      <mode name="full" requirements="OpenGL available" description="Full 3D rendering with meshes"/>
      <mode name="headless" requirements="None" description="Text-based world analysis and validation"/>
      <mode name="fallback" requirements="PIL/matplotlib" description="2D visualization and testing"/>
    </operation_modes>
    <locking scope="project,run" timeout_s="30" backoff="exponential"/>
    <mutation allow_paths="agents/**, renderer/**, scripts/**, forge/**, runs/**" no_exec_scripts="true"/>
  </non-negotiables>

  <environment os="Linux Mint" shell="bash" transport="stdio" alt_transport_port="5174"/>

  <visual_quality_standards>
    <mesh_integration>
      <standard>Adjacent chunk edges connect seamlessly without gaps or overlaps; no T-junctions.</standard>
      <standard>Materials transition smoothly across chunk boundaries; consistent UVs.</standard>
    </mesh_integration>
    <shading_correctness>
      <standard>Normals/tangents conform to MikkTSpace orientation; correct handedness for normal maps.</standard>
      <standard>Bounds accurate for frustum culling; no popping due to incorrect radii.</standard>
    </shading_correctness>
  </visual_quality_standards>

  <terrain_pipeline_tasks order="strict" start_index="T02">
    <task id="T02">Cube-sphere primitive (single-res) in Agent D.</task>
    <task id="T03">Normals & tangents (CPU) + MikkTSpace hook.</task>
    <task id="T04">PCC terrain spec → heightfield module (no displacement yet).</task>
    <task id="T05">Apply heightfield displacement to cube-sphere.</task>
    <task id="T06">Per-face quadtree chunking (static LOD scaffold).</task>
    <task id="T07">Crack prevention between LOD levels (stitching preferred; skirts as fallback).</task>
    <task id="T08">Runtime LOD selection + frustum culling.</task>
    <task id="T09">SDF module & voxelization harness (for caves/overhangs).</task>
    <task id="T10">Marching Cubes polygonization for SDF chunks.</task>
    <task id="T11">Blend surface heightfield with SDF (fusion & seams).</task>
    <task id="T12">Material & tangent space correctness pass.</task>
    <task id="T13">Determinism & reproducibility harness.</task>
    <task id="T14">Performance pass (multithreaded bake + streaming).</task>
    <task id="T15">PCC schema hardening for terrain pipeline.</task>
    <task id="T16">Viewer tools: diagnostics HUD & toggles.</task>
    <task id="T17">Example “hero” planet content (authoring pass).</task>
    <task id="T18">Deferred: hooks for structure mode (WFC tiling only).</task>
  </terrain_pipeline_tasks>

  <notes_and_guardrails>
    <determinism>Every stochastic module has a required seed; record seed_used in all outputs.</determinism>
    <manifest_format><![CDATA[
{
  "mesh": {
    "primitive_topology": "triangles",
    "positions": "buffer://.../pos.bin",
    "normals": "buffer://.../nrm.bin",
    "tangents": "buffer://.../tan.bin",
    "uv0": "buffer://.../uv.bin",
    "indices": "buffer://.../idx.bin",
    "bounds": {"center":[x,y,z], "radius": r},
    "material": {"albedo":"...", "normal":"..."}
  }
}
    ]]></manifest_format>
    <crack_prevention>Prefer stitching; allow skirts as fallback.</crack_prevention>
    <seams>Duplicate one-voxel overlap for SDF chunks; reconcile isosurface.</seams>
    <viewer>Keep a simple PBR/Phong shader; verify tangent-space orientation.</viewer>
  </notes_and_guardrails>

  <opengl_independence>
    <fallback_modes>
      <mode name="text_analysis">Analyze world structure, validate placement, generate reports</mode>
      <mode name="2d_visualization">Generate top-down maps, cross-sections, asset placement diagrams</mode>
      <mode name="headless_testing">Full world generation and validation without rendering</mode>
    </fallback_modes>
    <error_handling>
      <standard>Clear diagnostic messages when OpenGL unavailable</standard>
      <standard>Automatic fallback mode selection based on available libraries</standard>
      <standard>Alternative testing paths that don't require 3D rendering</standard>
      <standard>Comprehensive logging for debugging without visual output</standard>
    </error_handling>
  </opengl_independence>

  <mcp-contract schemas_dir="schemas" io_schema="schemas/mcp_tool_io.schema.json">
    <!-- Reuse Week-4 tools; terrain pipeline may register additional tools via contracts section -->
    <tool name="meshes.generate_assets" in_schema="$id:tool.meshes.generate_assets.in" out_schema="$id:tool.meshes.generate_assets.out"/>
    <tool name="meshes.validate_consistency" in_schema="$id:tool.meshes.validate_consistency.in" out_schema="$id:tool.meshes.validate_consistency.out"/>
    <tool name="world.analyze_headless" in_schema="$id:tool.world.analyze_headless.in" out_schema="$id:tool.world.analyze_headless.out"/>
    <tool name="world.generate_2d_map" in_schema="$id:tool.world.generate_2d_map.in" out_schema="$id:tool.world.generate_2d_map.out"/>
    <tool name="navigation.test_spherical" in_schema="$id:tool.navigation.test_spherical.in" out_schema="$id:tool.navigation.test_spherical.out"/>
    <tool name="visual.validate_consistency" in_schema="$id:tool.visual.validate_consistency.in" out_schema="$id:tool.visual.validate_consistency.out"/>
    <tool name="fallback.detect_capabilities" in_schema="$id:tool.fallback.detect_capabilities.in" out_schema="$id:tool.fallback.detect_capabilities.out"/>
    <error-model code_enum="opengl_unavailable,mesh_generation_failed,asset_inconsistency,navigation_error,fallback_required,NOT_IMPLEMENTED"/>
  </mcp-contract>

  <schemas>
    <schema id="schemas/mesh_generation.v1.schema.json" purpose="Asset generation requests and validation"/>
    <schema id="schemas/visual_consistency.v1.schema.json" purpose="Material and edge connection validation"/>
    <schema id="schemas/navigation_testing.v1.schema.json" purpose="Spherical navigation validation"/>
    <schema id="schemas/fallback_detection.v1.schema.json" purpose="Capability detection and mode selection"/>
    <schema id="schemas/world_analysis.v1.schema.json" purpose="Headless world structure analysis"/>
    <schema id="schemas/2d_visualization.v1.schema.json" purpose="Non-OpenGL visual representation"/>
  </schemas>

  <artifacts_spec>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/meshes_generated.json" when="meshes.generate_assets"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/visual_consistency.json" when="meshes.validate_consistency"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/world_analysis.json" when="world.analyze_headless"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/world_map_2d.png" when="world.generate_2d_map"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/navigation_test.json" when="navigation.test_spherical"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/capabilities.json" when="fallback.detect_capabilities"/>
  </artifacts_spec>

  <performance_targets>
    <terrain_pipeline>
      <target name="single_res_cube_sphere_build" value="<50ms for 128x128 per face" priority="high"/>
      <target name="lod_switch_popping" value="imperceptible with stitching/skirts" priority="critical"/>
      <target name="streaming_budget" value=">60fps camera fly with chunk streaming" priority="high"/>
      <target name="sdf_chunk_march_time" value="<100ms for 32^3 grid" priority="medium"/>
    </terrain_pipeline>
  </performance_targets>

  <definition-of-done>
    <item>Cube-sphere with correct normals/tangents and heightfield displacement.</item>
    <item>Per-face quadtree chunking with crack-free LOD transitions and frustum-culling-driven runtime selection.</item>
    <item>SDF module with voxelization + Marching Cubes, blended with surface heightfield without visible seams.</item>
    <item>Deterministic outputs with seeds recorded; performance harness shows budget adherence; viewer HUD/toggles implemented.</item>
    <item>Example “hero” planet produced and committed as artifacts.</item>
  </definition-of-done>

  <response-style>
    <format>PLAN → ACTIONS → VERIFY → RESULTS → NEXT</format>
    <verbosity>terse, command-first, comprehensive testing</verbosity>
    <disallowed>cloud calls, non-MCP I/O, OpenGL assumptions</disallowed>
  </response-style>

  <!-- HOP:CONTRACTS v1 START -->
  <enhanced-contracts>
    <invariants>
      <principle>Additive only (no deletions)</principle>
      <principle>Idempotent anchors throughout system</principle>
      <principle>Schema hygiene: include $schema, $id, version in new schemas</principle>
      <principle>Stub error contract: {"success":false,"error":{"code":"NOT_IMPLEMENTED","message":"","details":{}}}</principle>
      <principle>run_id = {UTC_ISO}--{GLOBAL_SEED}; artifacts under runs/{run_id}/...</principle>
      <principle>run_manifest.json must include: run_id, utc_iso, global_seed, inputs_hash, provenance{parent_ids,author,remix_splits}, tools_used[]</principle>
      <principle>Seed cascade: global_seed → world_seed → tool_seed → chip_seed; all tool outputs echo seed_used</principle>
      <principle>Tool namespaces allowed: tools.validators.*, tools.generators.*, tools.builders.*</principle>
    </invariants>

    <contract id="blueprint_chips">
      <tools>
        <tool name="tools.generators.blueprints.emit_chip" in_schema="$id:tool.generators.blueprints.emit_chip.in" out_schema="$id:tool.generators.blueprints.emit_chip.out"/>
        <tool name="tools.generators.blueprints.import_chip" in_schema="$id:tool.generators.blueprints.import_chip.in" out_schema="$id:tool.generators.blueprints.import_chip.out"/>
      </tools>
      <schema id="schemas/blueprint_chip.v1.schema.json" purpose="Prompt -> deterministic node graph chip"/>
      <artifacts path="runs/{run_id}/chips/"/>
    </contract>

    <contract id="replay_feedback">
      <tools>
        <tool name="tools.validators.replay_record" in_schema="$id:tool.validators.replay_record.in" out_schema="$id:tool.validators.replay_record.out"/>
        <tool name="tools.validators.replay_play" in_schema="$id:tool.validators.replay_play.in" out_schema="$id:tool.validators.replay_play.out"/>
      </tools>
      <schema id="schemas/replay.v1.schema.json" purpose="Record/play exact sim seeds and inputs"/>
      <artifacts path="runs/{run_id}/replays/"/>
    </contract>

    <contract id="performance_pass">
      <tools>
        <tool name="tools.validators.performance_pass" in_schema="$id:tool.validators.performance_pass.in" out_schema="$id:tool.validators.performance_pass.out"/>
      </tools>
      <schema id="schemas/performance_pass.v1.schema.json" purpose="Budget metrics and badge output"/>
      <artifacts path="runs/{run_id}/performance_pass.json"/>
    </contract>

    <contract id="schematic_cards">
      <tools>
        <tool name="tools.generators.export_schematic" in_schema="$id:tool.generators.export_schematic.in" out_schema="$id:tool.generators.export_schematic.out"/>
        <tool name="tools.generators.import_schematic" in_schema="$id:tool.generators.import_schematic.in" out_schema="$id:tool.generators.import_schematic.out"/>
      </tools>
      <schema id="schemas/schematic_card.v1.schema.json" purpose="Self-contained build card with IO pins"/>
      <artifacts path="runs/{run_id}/schematics/"/>
    </contract>

    <contract id="diffable_worlds">
      <tools>
        <tool name="tools.validators.world_diff" in_schema="$id:tool.validators.world_diff.in" out_schema="$id:tool.validators.world_diff.out"/>
      </tools>
      <schema id="schemas/world_diff.v1.schema.json" purpose="Object/graph diff format with stable IDs"/>
      <artifacts path="runs/{run_id}/diff.json"/>
    </contract>

    <contract id="deterministic_seeds">
      <documentation>All tools must emit seed_used and inputs_hash and follow the cascade: global_seed → world_seed → tool_seed → chip_seed</documentation>
    </contract>

    <contract id="provenance_remix">
      <schema id="schemas/provenance.v1.schema.json" purpose="Ancestry graph, authorship, remix credits/splits"/>
      <requirement>run_manifest.json must include: provenance.parent_ids, provenance.author, provenance.remix_splits</requirement>
    </contract>

    <contract id="safety_baseline">
      <tools>
        <tool name="tools.validators.safety_tag" in_schema="$id:tool.validators.safety_tag.in" out_schema="$id:tool.validators.safety_tag.out"/>
      </tools>
      <schema id="schemas/safety_tag.v1.schema.json" purpose="Age/comfort tags and default-private policy"/>
      <policy>Default visibility private unless safety_tag.ok===true</policy>
    </contract>

    <contract id="undo_ledger">
      <tools>
        <tool name="tools.builders.ledger_write" in_schema="$id:tool.builders.ledger_write.in" out_schema="$id:tool.builders.ledger_write.out"/>
      </tools>
      <schema id="schemas/ledger.v1.schema.json" purpose="Undo/redo time-travel entries across sessions"/>
      <artifacts path="runs/{run_id}/ledger.json"/>
    </contract>

    <contract id="building_system">
      <tools>
        <tool name="tools.builders.load_system" in_schema="$id:tool.builders.load_system.in" out_schema="$id:tool.builders.load_system.out"/>
        <tool name="tools.builders.validate_placement" in_schema="$id:tool.builders.validate_placement.in" out_schema="$id:tool.builders.validate_placement.out"/>
        <tool name="tools.builders.apply_placement" in_schema="$id:tool.builders.apply_placement.in" out_schema="$id:tool.builders.apply_placement.out"/>
      </tools>
      <schema id="schemas/building_system.v1.schema.json" purpose="JSON-Schema mirror of XML spec: Grid/Subgrid/Networks/Catalog/Blueprints/Prefabs/Creation"/>
      <artifacts path="runs/{run_id}/building_system.json"/>
    </contract>
  </enhanced-contracts>
  <!-- HOP:CONTRACTS v1 END -->
</high-order-prompt>
```


