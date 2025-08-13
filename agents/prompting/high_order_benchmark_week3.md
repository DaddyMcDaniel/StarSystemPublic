Summary: Week-3 High-Order Prompt (governor). End-to-end loop with gl runtime default, miniplanet seed + 3D first-person navigation, and robust Building System MVP. MCP-only; deterministic; local-first.

```xml
<high-order-prompt version="3.0"
 model="claude sonnet 4, unless specified otherwise in Low-order prompt"
 role="governor">
  <!-- Self-contained Week-3 benchmark. Do not rely on external text. -->
  <project id="starsystem" name="StarSystem + The Forge">
    <phase week="3" name="Orchestrated loop + alpha proof (miniplanet + FP nav + building MVP)" seed="${GLOBAL_SEED}" timezone="UTC"/>
    <vision>
      <north_star>Idea → playable mini-planet → shareable link in <10 minutes.</north_star>
      <alpha_proof>Generate a mini-planet seed, run it with proper 3D first-person navigation, and provide a robust Building System MVP.</alpha_proof>
      <runtime_policy>Default to OpenGL viewer (renderer/pcc_simple_viewer.py). Keep Godot headless behind a feature flag and return deterministic stubs when missing.</runtime_policy>
    </vision>

    <goals>
      <primary>
        <goal>Deliver a seamless local loop (Worldsmith → Vision-QA → Fixer) callable via scripts; artifacts under runs/{UTC_ISO}--{GLOBAL_SEED}.</goal>
        <goal>Implement miniplanet seed generation and load/run in OpenGL viewer with 3D first-person navigation (WASD + mouse-look).</goal>
        <goal>Building System MVP: grid/snap/subgrid/placement/undo with ledger persistence.</goal>
      </primary>
      <secondary>
        <goal>One functional network: Power (generator/cable/consumer) with solver + overlay.</goal>
        <goal>Deterministic replay + 0.5s time-scrub preview for placements.</goal>
        <goal>Observability HUD + Performance Pass script.</goal>
      </secondary>
    </goals>

    <scope include="renderer/pcc_simple_viewer.py, forge/core, forge/modules, mcp_server/server.py, schemas, scripts, config, tests, runs/ tooling"
           exclude="cloud LLMs, multiplayer netcode, asset stores, export pipelines, VR runtimes"/>
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
    <versions python="3.10..3.12" godot="4.x headless (optional)"/>
    <discovery godot_order="godot4-headless,godot4,godot,./godot.AppImage --headless" cache="./.cache/godot_path"/>
    <locking scope="project,run" timeout_s="30" backoff="exponential"/>
    <mutation allow_paths="examples/projects/*, forge/**, renderer/**, scripts/**" no_exec_scripts="true"/>
  </non-negotiables>

  <environment os="Linux Mint" shell="bash" transport="stdio" alt_transport_port="5174"/>

  <runtime>
    <toggle>FORGE_RUNTIME=gl|godot</toggle>
    <default>gl</default>
    <gl path="renderer/pcc_simple_viewer.py"/>
    <godot mode="headless" behind_flag="true" stub_when_missing="true"/>
    <acceptance>One command runs the world for gl; the godot path is disabled unless FORGE_RUNTIME=godot.</acceptance>
  </runtime>

  <mcp-contract schemas_dir="schemas" io_schema="schemas/mcp_tool_io.schema.json">
    <!-- Week-1/2 tools remain. Week-3 additions below. -->
    <!-- World hot-swap, layer listing/editing, preview microsim, power graph, HUD/perf pass, publish gatekeeper, mode switcher. -->
    <tool name="world.hot_swap" in_schema="$id:tool.world.hot_swap.in" out_schema="$id:tool.world.hot_swap.out"/>
    <tool name="world.list_layers" in_schema="$id:tool.world.list_layers.in" out_schema="$id:tool.world.list_layers.out"/>
    <tool name="world.edit_layer" in_schema="$id:tool.world.edit_layer.in" out_schema="$id:tool.world.edit_layer.out"/>
    <tool name="preview.microsim" in_schema="$id:tool.preview.microsim.in" out_schema="$id:tool.preview.microsim.out"/>
    <tool name="power.solve" in_schema="$id:tool.power.solve.in" out_schema="$id:tool.power.solve.out"/>
    <tool name="ui.hud_stats" in_schema="$id:tool.ui.hud_stats.in" out_schema="$id:tool.ui.hud_stats.out"/>
    <tool name="validators.performance_pass" in_schema="$id:tool.validators.performance_pass.in" out_schema="$id:tool.validators.performance_pass.out"/>
    <tool name="publish.dry_run" in_schema="$id:tool.publish.dry_run.in" out_schema="$id:tool.publish.dry_run.out"/>
    <!-- Building MVP (grid/snap/placement/undo) -->
    <tool name="builders.validate_placement" in_schema="$id:tool.builders.validate_placement.in" out_schema="$id:tool.builders.validate_placement.out"/>
    <tool name="builders.apply_placement" in_schema="$id:tool.builders.apply_placement.in" out_schema="$id:tool.builders.apply_placement.out"/>
    <tool name="ledger.write" in_schema="$id:tool.ledger.write.in" out_schema="$id:tool.ledger.write.out"/>

    <!-- HOP:CONTRACTS v1 START -->
    <!-- Preserve Week-1 anchored contracts here; Opus-1 may update this block idempotently. -->
    <!-- HOP:CONTRACTS v1 END -->

    <error-model code_enum="godot_missing,gate_violation,lock_timeout,timeout,budget_exceeded,invalid_schema,NOT_IMPLEMENTED"/>
  </mcp-contract>

  <schemas>
    <!-- Freeze at v1: reference Week-1/2 base and enhancements. Add Week-3: hot_swap, preview, power, publish, layers. -->
    <schema id="schemas/world_hot_swap.v1.schema.json" purpose="EditWorld→SimWorld op-log diff application"/>
    <schema id="schemas/preview_microsim.v1.schema.json" purpose="0.5s deterministic look-ahead for pending placement"/>
    <schema id="schemas/power_graph.v1.schema.json" purpose="Generators/cables/consumers + solver inputs/outputs"/>
    <schema id="schemas/publish_gate.v1.schema.json" purpose="Privacy/safety/perf/attributions gate report"/>
    <schema id="schemas/world_layers.v1.schema.json" purpose="Orbit/Surface/Subsurface layer listing and edit envelope"/>
  </schemas>

  <artifacts_spec>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/oplog.jsonl" when="world.hot_swap"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/replays/${GLOBAL_SEED}.jsonl" when="preview.microsim"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/power_overlay.json" when="power.solve"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/hud_stats.json" when="ui.hud_stats"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/performance_pass.json" when="validators.performance_pass"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/publish_report.json" when="publish.dry_run"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/ledger.jsonl" when="ledger.write"/>
  </artifacts_spec>

  <observability>
    <logs tools_jsonl="runs/.../logs/tools.jsonl"/>
    <manifest file="runs/.../run_manifest.json"/>
    <record>
      <field>run_id</field>
      <field>utc_iso</field>
      <field>global_seed</field>
      <field>inputs_hash</field>
      <field>seed_used</field>
      <field>provenance.parent_ids</field>
      <field>provenance.author</field>
      <field>provenance.remix_splits</field>
      <field>tools_used</field>
    </record>
  </observability>

  <gates profile_file="config/gates.web.yaml" alt_profile_file="config/gates.native.yaml">
    <hard_caps>
      <metric name="tri_count" value="120000"/>
      <metric name="materials" value="32"/>
    </hard_caps>
    <priorities order="readability,perf_proxy,tri_count"/>
    <tolerances>
      <tri_count worsen_max_pct="2.0"/>
      <materials worsen_max="1"/>
    </tolerances>
    <tie_breakers order="perf_proxy,tri_count"/>
    <nan_policy>treat_as_worst</nan_policy>
  </gates>

  <budgets calls_max="100" wall_seconds="900" per_call_timeout="120" fixer_max_iters="3" cancel_on_gate_violation="true"/>

  <definition-of-done>
    <item>OpenGL viewer runs a generated mini-planet seed with 3D first-person navigation (WASD + mouse-look) under FORGE_RUNTIME=gl.</item>
    <item>Building MVP: place/move/delete 500+ blocks with grid/snap/placement validator and undo/redo persisted in ledger.jsonl.</item>
    <item>Optional Godot path: MCP tools return deterministic stubs when missing; feature-flagged by FORGE_RUNTIME=godot.</item>
    <item>Power network overlay and preview microsim produce deterministic outputs; replays match bit-for-bit on repeated runs.</item>
    <item>HUD visible and Performance Pass emits badge to performance_pass.json; publish gate blocks until safety+perf pass green.</item>
  </definition-of-done>

  <response-style>
    <format>PLAN → ACTIONS → VERIFY → RESULTS → NEXT</format>
    <verbosity>terse, command-first, no hidden steps</verbosity>
    <disallowed>cloud calls, non-MCP I/O, interactive prompts without APPROVE_INSTALL=1</disallowed>
  </response-style>
</high-order-prompt>
```


