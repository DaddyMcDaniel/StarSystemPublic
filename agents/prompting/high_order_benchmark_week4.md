Summary: Week-4 High-Order Prompt (governor). Perfect world generation with seamless mesh integration, enhanced navigation systems, and robust OpenGL-independent operation. Visual appeal and consistency focus.

```xml
<high-order-prompt version="4.0"
 model="claude sonnet 4, unless specified otherwise in Low-order prompt"
 role="governor">
  <!-- Self-contained Week-4 benchmark. Do not rely on external text. -->
  <project id="starsystem" name="StarSystem + The Forge">
    <phase week="4" name="Perfect worldgen + seamless mesh pipeline + robust operation" seed="${GLOBAL_SEED}" timezone="UTC"/>
    <vision>
      <north_star>Idea → visually stunning mini-planet → seamless navigation → consistent assets → <10 minutes.</north_star>
      <visual_excellence>Mesh shapes connect seamlessly at edges, materials flow consistently, terrain feels organic and navigable.</visual_excellence>
      <operation_robustness>System operates flawlessly with or without OpenGL, providing meaningful fallbacks and clear error guidance.</operation_robustness>
    </vision>

    <goals>
      <primary>
        <goal>Complete mesh pipeline: asset generation, loading, and rendering with seamless edge connections and consistent materials.</goal>
        <goal>Perfect spherical world navigation: smooth surface walking, proper mouse capture, collision-free spawn zones.</goal>
        <goal>OpenGL-independent operation: robust fallbacks, clear error messages, alternative testing modes.</goal>
        <goal>Visual consistency: materials blend smoothly, terrain feels organic, assets match environmental themes.</goal>
      </primary>
      <secondary>
        <goal>Enhanced world generation: grounded structures, thematic coherence, varied but consistent landscapes.</goal>
        <goal>Robust testing pipeline: comprehensive validation without OpenGL dependency.</goal>
        <goal>Performance optimization: efficient mesh rendering, batched draws, memory management.</goal>
      </secondary>
    </goals>

    <scope include="mesh pipeline completion, OpenGL fallback systems, visual consistency, navigation perfection, world generation enhancement"
           exclude="multiplayer, cloud services, VR systems, external asset stores"/>
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
      <standard>Adjacent mesh edges connect seamlessly without gaps or overlaps</standard>
      <standard>Materials transition smoothly between connected assets</standard>
      <standard>Terrain-asset boundaries feel organic and natural</standard>
      <standard>Scale consistency maintained across all generated assets</standard>
    </mesh_integration>
    <navigation_excellence>
      <standard>Smooth spherical surface walking without jitter or sticking</standard>
      <standard>Mouse capture maintains infinite rotation without boundary limits</standard>
      <standard>Spawn positions always safe with adequate clearance</standard>
      <standard>Collision detection prevents clipping through objects</standard>
    </navigation_excellence>
    <world_generation>
      <standard>Structures are properly grounded on terrain surface</standard>
      <standard>Thematic consistency across all generated features</standard>
      <standard>Varied landscapes that feel coherent and purposeful</standard>
      <standard>Resource distribution supports intended gameplay</standard>
    </world_generation>
  </visual_quality_standards>

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
    <!-- Week-1/2/3 tools remain. Week-4 additions for mesh pipeline and fallbacks. -->
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
    <!-- Week-4 schema additions for mesh pipeline and fallback systems -->
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
    <mesh_rendering>
      <target name="mesh_load_time" value="<2s for 50 assets" priority="high"/>
      <target name="frame_rate" value=">30fps with 100+ meshes" priority="medium"/>
      <target name="memory_usage" value="<512MB for typical scene" priority="medium"/>
    </mesh_rendering>
    <world_generation>
      <target name="generation_time" value="<5s for full planet" priority="high"/>
      <target name="asset_consistency" value=">95% thematic match" priority="high"/>
      <target name="edge_connection" value="<1% gap/overlap ratio" priority="critical"/>
    </world_generation>
  </performance_targets>

  <definition-of-done>
    <item>Mesh pipeline complete: assets generate, load, render with seamless edge connections</item>
    <item>OpenGL-independent operation: all manual/auto loops work with meaningful fallbacks</item>
    <item>Perfect spherical navigation: smooth walking, infinite mouse rotation, safe spawning</item>
    <item>Visual consistency: materials flow smoothly, assets match environmental themes</item>
    <item>Comprehensive testing: validation works without OpenGL dependency</item>
    <item>Performance targets met: fast generation, smooth rendering, efficient memory use</item>
  </definition-of-done>

  <response-style>
    <format>PLAN → ACTIONS → VERIFY → RESULTS → NEXT</format>
    <verbosity>terse, command-first, comprehensive testing</verbosity>
    <disallowed>cloud calls, non-MCP I/O, OpenGL assumptions</disallowed>
  </response-style>
</high-order-prompt>
```