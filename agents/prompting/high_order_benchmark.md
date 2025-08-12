Summary: Vision-aligned canonical high-order benchmark prompt (governor). All low-order prompts and actions must obey this.

```xml
<high-order-prompt version="1.0" 
model= "claude sonnet 4, unless specified otherwise in Low-order prompt" 
role="governor">
  <!-- Fill this once at the start of the week/phase. Keep it self-contained. -->
  <project id="starsystem" name="StarSystem + The Forge">
    <phase week="1" name="Pre-alpha Core (MCP + headless hooks + planet/nav stubs)" seed="${GLOBAL_SEED}" timezone="UTC"/>
    <vision>
      <!-- Product vision drives all technical choices -->
      <hub name="StarSystem" description="Cross-platform 3D hub (web/desktop/VR) that launches player-made game apps, like Roblox but AI-assisted and Godot-based."/>
      <creator name="The Forge" description="Launchable in-world creation tool; same engine, separate seed; deterministic build mode toggled on demand."/>
      <worlds type="mini-planet sandbox" inspiration="Eco, Astroneer, Halo Forge" note="Gameplay is goal-lite, focused on creative building; build mode lands after planet generation + navigation are solid."/>
      <ai_integration>Local-only agents orchestrate generation/QA/fixing via MCP tools; no cloud keys.</ai_integration>
      <engineering edge="Efficient headless Godot 4 + deterministic mesh/proc-gen + strict schemas/gates."/>
    </vision>

    <goals>
      <primary>
        <goal>Ship a deterministic, MCP-only Python server exposing headless Godot utilities and local generators/validators.</goal>
        <goal>Create repo scaffolding, strict schemas, tests, and runs/ artifact discipline.</goal>
        <goal>Implement minimal planet layout generator + nav feasibility checks (stub ok if Godot missing).</goal>
        <goal>Provide a tiny hub scene & CLI smoke path to validate loop: generate → apply_patch → test_headless → capture_views → dump_scene.</goal>
      </primary>
      <secondary>
        <goal>Document Claude Code ↔ MCP hookup (stdio/WS) and local discovery of Godot.</goal>
        <goal>Lay placeholders for Week-2 gates/feedback and Week-3 loop.</goal>
      </secondary>
    </goals>

    <scope include="mcp_server, schemas, godot headless scripts, generators(maze/planet), validators, examples/min_scene, tests, runs/ tooling"
           exclude="multiplayer netcode, in-game build UI, cloud services, asset stores, export pipelines, VR runtimes (defer to later week)"/>
  </project>

  <non-negotiables>
    <interface>MCP-only (official Python SDK); tools carry explicit input/output $id schemas.</interface>
    <cloud-llms>forbidden</cloud-llms>
    <secrets>do-not-read-or-require</secrets>
    <determinism rng="pcg64">
      <seed-cascade>tool_seed = hash(tool_name, GLOBAL_SEED, frozen_inputs)</seed-cascade>
      <float-tolerance>1e-9</float-tolerance>
      <time-source>UTC only; no wallclock in logic</time-source>
    </determinism>
    <artifacts dir="./runs/${UTC_ISO}-${GLOBAL_SEED}" latest_symlink="./runs/latest" manifest="run_manifest.json"/>
    <installs approval_flag="APPROVE_INSTALL" behavior="print-plan-and-exit-if-unset"/>
    <versions python="3.10..3.12" godot="4.x headless pinned in README"/>
    <discovery godot_order="godot4-headless,godot4,godot,./godot.AppImage --headless" cache="./.cache/godot_path"/>
    <locking scope="project,run" timeout_s="30" backoff="exponential"/>
    <mutation allow_paths="examples/projects/*" no_exec_scripts="true"/>
  </non-negotiables>

  <environment os="Linux Mint" shell="bash" transport="stdio" alt_transport_port="5174"/>

  <mcp-contract schemas_dir="schemas" io_schema="schemas/mcp_tool_io.schema.json">
    <tool name="godot.test_headless"      in_schema="$id:tool.godot.test_headless.in" out_schema="$id:tool.godot.test_headless.out"/>
    <tool name="godot.capture_views"      in_schema="$id:tool.godot.capture_views.in" out_schema="$id:tool.godot.capture_views.out"/>
    <tool name="godot.dump_scene"         in_schema="$id:tool.godot.dump_scene.in" out_schema="$id:tool.godot.dump_scene.out"/>
    <tool name="godot.apply_patch"        in_schema="schemas/patch_plan.schema.json"   out_schema="$id:tool.godot.apply_patch.out"/>
    <tool name="generators.maze_generate" in_schema="$id:tool.generators.maze_generate.in" out_schema="$id:tool.generators.maze_generate.out"/>
    <tool name="validators.maze_graph_validate" in_schema="$id:tool.validators.maze_graph_validate.in" out_schema="$id:tool.validators.maze_graph_validate.out"/>
    <tool name="memory.search"            in_schema="$id:tool.memory.search.in" out_schema="$id:tool.memory.search.out"/>
    <!-- New tools for Week-1/2 scaffolding of selected enhancements -->
    <!-- Blueprint Chips (prompt -> deterministic node graphs) -->
    <tool name="blueprints.emit_chip"      in_schema="$id:tool.blueprints.emit_chip.in" out_schema="$id:tool.blueprints.emit_chip.out"/>
    <tool name="blueprints.import_chip"    in_schema="$id:tool.blueprints.import_chip.in" out_schema="$id:tool.blueprints.import_chip.out"/>
    <!-- Replay-first feedback (record/play with exact seeds) -->
    <tool name="validators.replay_record"  in_schema="$id:tool.validators.replay_record.in" out_schema="$id:tool.validators.replay_record.out"/>
    <tool name="validators.replay_play"    in_schema="$id:tool.validators.replay_play.in" out_schema="$id:tool.validators.replay_play.out"/>
    <!-- Performance Pass (budgets -> badge) -->
    <tool name="validators.performance_pass" in_schema="$id:tool.validators.performance_pass.in" out_schema="$id:tool.validators.performance_pass.out"/>
    <!-- Schematic Cards (export/import self-contained builds) -->
    <tool name="generators.export_schematic" in_schema="$id:tool.generators.export_schematic.in" out_schema="$id:tool.generators.export_schematic.out"/>
    <tool name="generators.import_schematic" in_schema="$id:tool.generators.import_schematic.in" out_schema="$id:tool.generators.import_schematic.out"/>
    <!-- Diffable Worlds (visual + graph diffs) -->
    <tool name="validators.world_diff"     in_schema="$id:tool.validators.world_diff.in" out_schema="$id:tool.validators.world_diff.out"/>
    <!-- Safety baseline tagging -->
    <tool name="validators.safety_tag"     in_schema="$id:tool.validators.safety_tag.in" out_schema="$id:tool.validators.safety_tag.out"/>
    <!-- Undo across sessions (time-travel ledger) -->
    <tool name="ledger.write"              in_schema="$id:tool.ledger.write.in" out_schema="$id:tool.ledger.write.out"/>
    <!-- Building System (grid/subgrid/networks/catalog) -->
    <tool name="builders.load_system"      in_schema="$id:tool.builders.load_system.in" out_schema="$id:tool.builders.load_system.out"/>
    <tool name="builders.validate_placement" in_schema="$id:tool.builders.validate_placement.in" out_schema="$id:tool.builders.validate_placement.out"/>
    <tool name="builders.apply_placement"  in_schema="$id:tool.builders.apply_placement.in" out_schema="$id:tool.builders.apply_placement.out"/>
    <error-model code_enum="godot_missing,gate_violation,lock_timeout,timeout,budget_exceeded,invalid_schema"/>
  </mcp-contract>

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

  <!-- Enhancements: Schemas and Artifacts (required even if tools stub) -->
  <schemas>
    <schema id="schemas/blueprint_chip.schema.json" purpose="Prompt -> deterministic node graph chip"/>
    <schema id="schemas/replay.schema.json"        purpose="Record/play exact sim seeds and inputs"/>
    <schema id="schemas/performance_pass.schema.json" purpose="Budget metrics and badge output"/>
    <schema id="schemas/schematic_card.schema.json" purpose="Self-contained build card with IO pins"/>
    <schema id="schemas/world_diff.schema.json"    purpose="Object/graph diff format with stable IDs"/>
    <schema id="schemas/provenance.schema.json"    purpose="Ancestry graph, authorship, remix credits/splits"/>
    <schema id="schemas/safety_tag.schema.json"    purpose="Age/comfort tags and default-private policy"/>
    <schema id="schemas/ledger.schema.json"        purpose="Undo/redo time-travel entries across sessions"/>
    <schema id="schemas/building_system.schema.json" purpose="Grid rules, networks, catalog, blueprints (see BuildingSystem v1.0)"/>
  </schemas>

  <artifacts_spec>
    <dir path="runs/${UTC_ISO}-${GLOBAL_SEED}/chips/" when="blueprints.emit_chip"/>
    <dir path="runs/${UTC_ISO}-${GLOBAL_SEED}/replays/" when="validators.replay_record"/>
    <file path="runs/${UTC_ISO}-${GLOBAL_SEED}/performance_pass.json" when="validators.performance_pass"/>
    <dir path="runs/${UTC_ISO}-${GLOBAL_SEED}/schematics/" when="generators.export_schematic"/>
    <file path="runs/${UTC_ISO}-${GLOBAL_SEED}/diff.json" when="validators.world_diff"/>
    <file path="runs/${UTC_ISO}-${GLOBAL_SEED}/ledger.json" when="ledger.write"/>
    <file path="runs/${UTC_ISO}-${GLOBAL_SEED}/building_system.json" when="builders.load_system"/>
  </artifacts_spec>

  <observability>
    <logs tools_jsonl="runs/.../logs/tools.jsonl"/>
    <manifest file="runs/.../run_manifest.json"/>
    <!-- Determinism + provenance logging -->
    <record>
      <field>seed_used</field>
      <field>inputs_hash</field>
      <field>provenance.parent_ids</field>
      <field>provenance.author</field>
      <field>provenance.remix_splits</field>
    </record>
  </observability>

  <definition-of-done>
    <item>MCP server boots; tool manifest prints with $id schemas.</item>
    <item>`make smoke` executes: generate → apply_patch(dry_run→apply) → test_headless → dump_scene → capture_views; artifacts under ./runs/… and ./runs/latest updated atomically.</item>
    <item>Tests pass; if Godot missing, stub tools return structured `godot_missing` without crash, tests skip accordingly.</item>
    <item>README shows Claude Code client config (stdio/WS) and Godot discovery order.</item>
  </definition-of-done>

  <response-style>
    <format>PLAN → ACTIONS → VERIFY → RESULTS → NEXT</format>
    <verbosity>terse, command-first, no hidden steps</verbosity>
    <disallowed>cloud calls, non-MCP I/O, interactive prompts without APPROVE_INSTALL=1</disallowed>
  </response-style>

  <!-- HOP:CONTRACTS v1 START -->
  <enhanced-contracts>
    <!-- Global invariants for all contracts -->
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

    <!-- Contract 1: Blueprint Chips -->
    <contract id="blueprint_chips">
      <tools>
        <tool name="tools.generators.blueprints.emit_chip" in_schema="$id:tool.generators.blueprints.emit_chip.in" out_schema="$id:tool.generators.blueprints.emit_chip.out"/>
        <tool name="tools.generators.blueprints.import_chip" in_schema="$id:tool.generators.blueprints.import_chip.in" out_schema="$id:tool.generators.blueprints.import_chip.out"/>
      </tools>
      <schema id="schemas/blueprint_chip.v1.schema.json" purpose="Prompt -> deterministic node graph chip"/>
      <artifacts path="runs/{run_id}/chips/"/>
    </contract>

    <!-- Contract 2: Replay-first feedback -->
    <contract id="replay_feedback">
      <tools>
        <tool name="tools.validators.replay_record" in_schema="$id:tool.validators.replay_record.in" out_schema="$id:tool.validators.replay_record.out"/>
        <tool name="tools.validators.replay_play" in_schema="$id:tool.validators.replay_play.in" out_schema="$id:tool.validators.replay_play.out"/>
      </tools>
      <schema id="schemas/replay.v1.schema.json" purpose="Record/play exact sim seeds and inputs"/>
      <artifacts path="runs/{run_id}/replays/"/>
    </contract>

    <!-- Contract 3: Performance Pass -->
    <contract id="performance_pass">
      <tools>
        <tool name="tools.validators.performance_pass" in_schema="$id:tool.validators.performance_pass.in" out_schema="$id:tool.validators.performance_pass.out"/>
      </tools>
      <schema id="schemas/performance_pass.v1.schema.json" purpose="Budget metrics and badge output"/>
      <artifacts path="runs/{run_id}/performance_pass.json"/>
    </contract>

    <!-- Contract 4: Schematic Cards -->
    <contract id="schematic_cards">
      <tools>
        <tool name="tools.generators.export_schematic" in_schema="$id:tool.generators.export_schematic.in" out_schema="$id:tool.generators.export_schematic.out"/>
        <tool name="tools.generators.import_schematic" in_schema="$id:tool.generators.import_schematic.in" out_schema="$id:tool.generators.import_schematic.out"/>
      </tools>
      <schema id="schemas/schematic_card.v1.schema.json" purpose="Self-contained build card with IO pins"/>
      <artifacts path="runs/{run_id}/schematics/"/>
    </contract>

    <!-- Contract 5: Diffable Worlds -->
    <contract id="diffable_worlds">
      <tools>
        <tool name="tools.validators.world_diff" in_schema="$id:tool.validators.world_diff.in" out_schema="$id:tool.validators.world_diff.out"/>
      </tools>
      <schema id="schemas/world_diff.v1.schema.json" purpose="Object/graph diff format with stable IDs"/>
      <artifacts path="runs/{run_id}/diff.json"/>
    </contract>

    <!-- Contract 6: Deterministic Seeds Everywhere -->
    <contract id="deterministic_seeds">
      <documentation>All tools must emit seed_used and inputs_hash and follow the cascade: global_seed → world_seed → tool_seed → chip_seed</documentation>
    </contract>

    <!-- Contract 7: Provenance / Remix Credits -->
    <contract id="provenance_remix">
      <schema id="schemas/provenance.v1.schema.json" purpose="Ancestry graph, authorship, remix credits/splits"/>
      <requirement>run_manifest.json must include: provenance.parent_ids, provenance.author, provenance.remix_splits</requirement>
    </contract>

    <!-- Contract 8: Safety Baseline -->
    <contract id="safety_baseline">
      <tools>
        <tool name="tools.validators.safety_tag" in_schema="$id:tool.validators.safety_tag.in" out_schema="$id:tool.validators.safety_tag.out"/>
      </tools>
      <schema id="schemas/safety_tag.v1.schema.json" purpose="Age/comfort tags and default-private policy"/>
      <policy>Default visibility private unless safety_tag.ok===true</policy>
    </contract>

    <!-- Contract 9: Undo Ledger -->
    <contract id="undo_ledger">
      <tools>
        <tool name="tools.builders.ledger_write" in_schema="$id:tool.builders.ledger_write.in" out_schema="$id:tool.builders.ledger_write.out"/>
      </tools>
      <schema id="schemas/ledger.v1.schema.json" purpose="Undo/redo time-travel entries across sessions"/>
      <artifacts path="runs/{run_id}/ledger.json"/>
    </contract>

    <!-- Contract 10: Building System v1.0 -->
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

#proposed additions:
    prompt created by: GPT-5 implement into HOP with: (Opus-1)
Role: Edit contracts only (schemas/tools/artifacts), then switch back to Claude 4 Sonnet for implementations.

Scope:
- Apply to hop_main: agents/prompting/high_order_benchmark.md.
- Mirror the same anchored section into all agent HOPs (no bodies, contracts only).
- Additive only; idempotent via anchors.

Global invariants:
- Additive only (no deletions).
- Idempotent anchors: <!-- HOP:CONTRACTS v1 START --> ... <!-- HOP:CONTRACTS v1 END -->
- Schema hygiene: include $schema, $id, version in new schemas.
- Stub error contract: {"success":false,"error":{"code":"NOT_IMPLEMENTED","message":"","details":{}}}
- run_id = {UTC_ISO}--{GLOBAL_SEED}; artifacts under runs/{run_id}/...
- run_manifest.json must include: run_id, utc_iso, global_seed, inputs_hash, provenance{parent_ids,author,remix_splits}, tools_used[]
- Seed cascade: global_seed → world_seed → tool_seed → chip_seed; all tool outputs echo seed_used.
- Tool namespaces allowed: tools.validators.*, tools.generators.*, tools.builders.*

Contracts to add (10):
1) Blueprint Chips
  - Tools: tools.generators.blueprints.emit_chip, tools.generators.blueprints.import_chip
  - Schema: schemas/blueprint_chip.v1.schema.json
  - Artifacts: runs/{run_id}/chips/

2) Replay-first feedback
  - Tools: tools.validators.replay_record, tools.validators.replay_play
  - Schema: schemas/replay.v1.schema.json
  - Artifacts: runs/{run_id}/replays/

3) Performance Pass
  - Tool: tools.validators.performance_pass
  - Schema: schemas/performance_pass.v1.schema.json
  - Artifact: runs/{run_id}/performance_pass.json

4) Schematic Cards
  - Tools: tools.generators.export_schematic, tools.generators.import_schematic
  - Schema: schemas/schematic_card.v1.schema.json
  - Artifacts: runs/{run_id}/schematics/

5) Diffable Worlds
  - Tool: tools.validators.world_diff
  - Schema: schemas/world_diff.v1.schema.json
  - Artifact: runs/{run_id}/diff.json

6) Deterministic Seeds Everywhere
  - Document in hop_main: all tools must emit seed_used and inputs_hash and follow the cascade.

7) Provenance / Remix Credits
  - Schema: schemas/provenance.v1.schema.json
  - Require in run_manifest.json: provenance.parent_ids, provenance.author, provenance.remix_splits

8) Safety Baseline
  - Tool: tools.validators.safety_tag
  - Schema: schemas/safety_tag.v1.schema.json
  - Policy: default visibility private unless safety_tag.ok===true

9) Undo Ledger
  - Tool: tools.builders.ledger_write
  - Schema: schemas/ledger.v1.schema.json
  - Artifact: runs/{run_id}/ledger.json

10) Building System v1.0
  - Schema: schemas/building_system.v1.schema.json (JSON-Schema mirror of XML spec: Grid/Subgrid/Networks/Catalog/Blueprints/Prefabs/Creation)
  - Tools: tools.builders.load_system, tools.builders.validate_placement, tools.builders.apply_placement
  - Artifact: runs/{run_id}/building_system.json

Deliverable:
- Insert one anchored section with all registrations (tools), schema $ids, artifact paths, observability notes, and stub error contract into hop_main and agent HOPs.
- Print a one-paragraph “done” summary with counts of tools/schemas added and the anchors used.
- Then SWITCH MODEL back to Claude 4 Sonnet for implementation work (schemas + tool bodies).
