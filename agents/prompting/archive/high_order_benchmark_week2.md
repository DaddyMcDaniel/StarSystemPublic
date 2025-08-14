Summary: Week-2 High-Order Prompt (governor). Schemas locked, gates enforced, human feedback CLI, camera & stats QA. MCP-only, no cloud keys. Aligns with StarSystem + The Forge vision and Week-1 contracts.

```xml
<high-order-prompt version="2.0"
 model="claude sonnet 4, unless specified otherwise in Low-order prompt"
 role="governor">
  <!-- Self-contained Week-2 benchmark. Do not rely on external text. -->
  <project id="starsystem" name="StarSystem + The Forge">
    <phase week="2" name="Schemas locked, gates enforced, human feedback, camera & stats QA" seed="${GLOBAL_SEED}" timezone="UTC"/>
    <vision>
      <north_star>Idea → playable mini-planet → shareable link in &lt;10 minutes.</north_star>
      <principles>Deterministic, local-first, MCP-only; prompt → Blueprint Chips; provenance-first remix culture.</principles>
    </vision>

    <goals>
      <primary>
        <goal>Freeze JSON Schemas with $schema, $id, version for all Week-1/2 contracts (no breaking changes this week).</goal>
        <goal>Enforce acceptance gates and non-regression validators; fail deterministically with rationale.</goal>
        <goal>Ship Human Feedback CLI with rubric (scores 1–5) and issues evidence; save human_feedback.json.</goal>
        <goal>Harden capture pipeline: deterministic camera orbits, modes_realized reporting, never crash on missing modes.</goal>
        <goal>Maintain deterministic artifacts under runs/{UTC_ISO}--{GLOBAL_SEED} and update runs/latest atomically.</goal>
      </primary>
      <secondary>
        <goal>Document schema freeze and MCP client hookup (stdio/WS) in README.</goal>
        <goal>Expand tests: schema validation of examples, gates violations, rollback/logging invariants.</goal>
      </secondary>
    </goals>

    <scope include="schemas, validators, human_feedback CLI, examples, tests, runs/ tooling"
           exclude="cloud LLMs, multiplayer, export pipelines, VR runtimes, asset stores"/>
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
    <versions python="3.10..3.12" godot="4.x headless pinned in README"/>
    <discovery godot_order="godot4-headless,godot4,godot,./godot.AppImage --headless" cache="./.cache/godot_path"/>
    <locking scope="project,run" timeout_s="30" backoff="exponential"/>
    <mutation allow_paths="examples/projects/*" no_exec_scripts="true"/>
  </non-negotiables>

  <environment os="Linux Mint" shell="bash" transport="stdio" alt_transport_port="5174"/>

  <mcp-contract schemas_dir="schemas" io_schema="schemas/mcp_tool_io.schema.json">
    <!-- Week-1 tools remain registered (unchanged). Add Week-2 validators. -->
    <tool name="validators.gates_check" in_schema="$id:tool.validators.gates_check.in" out_schema="$id:tool.validators.gates_check.out"/>
    <tool name="validators.non_regression" in_schema="$id:tool.validators.non_regression.in" out_schema="$id:tool.validators.non_regression.out"/>

    <!-- Week-1 enhancements remain available (Blueprint Chips, Replay, Performance Pass, Schematic Cards, Diffable Worlds, Safety, Ledger, Building System). -->

    <!-- HOP:CONTRACTS v1 START -->
    <!-- Preserve Week-1 anchored contracts here; Opus-1 may update this block idempotently. -->
    <!-- HOP:CONTRACTS v1 END -->

    <error-model code_enum="godot_missing,gate_violation,lock_timeout,timeout,budget_exceeded,invalid_schema,NOT_IMPLEMENTED"/>
  </mcp-contract>

  <schemas>
    <!-- Freeze these with $schema, $id, version (no breaking changes in Week-2). -->
    <schema id="schemas/patch_plan.v1.schema.json" purpose="Strict patch plan mini-grammar"/>
    <schema id="schemas/mcp_tool_io.schema.json" purpose="Base I/O envelope"/>
    <schema id="schemas/gates.v1.schema.json" purpose="Acceptance gates profiles and caps"/>
    <schema id="schemas/non_regression.v1.schema.json" purpose="Non-regression tolerances and rationale"/>
    <schema id="schemas/human_feedback.v1.schema.json" purpose="CLI rubric and issues evidence"/>
    <!-- From Week-1 enhancements (frozen at v1): -->
    <schema id="schemas/blueprint_chip.v1.schema.json" purpose="Prompt → deterministic node graph chip"/>
    <schema id="schemas/replay.v1.schema.json" purpose="Record/play exact sim seeds and inputs"/>
    <schema id="schemas/performance_pass.v1.schema.json" purpose="Budget metrics and badge output"/>
    <schema id="schemas/schematic_card.v1.schema.json" purpose="Self-contained build card with IO pins"/>
    <schema id="schemas/world_diff.v1.schema.json" purpose="Object/graph diff format with stable IDs"/>
    <schema id="schemas/provenance.v1.schema.json" purpose="Ancestry graph, authorship, remix credits/splits"/>
    <schema id="schemas/safety_tag.v1.schema.json" purpose="Age/comfort tags and default-private policy"/>
    <schema id="schemas/ledger.v1.schema.json" purpose="Undo/redo time-travel entries across sessions"/>
    <schema id="schemas/building_system.v1.schema.json" purpose="Grid/Subgrid/Networks/Catalog/Blueprints/Prefabs/Creation"/>
  </schemas>

  <artifacts_spec>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/human_feedback.json" when="human_feedback_cli"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/captures_meta.json" when="godot.capture_views"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/scene_stats.json" when="godot.dump_scene"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/gates_report.json" when="validators.gates_check"/>
    <file path="runs/${UTC_ISO}--${GLOBAL_SEED}/non_regression.json" when="validators.non_regression"/>
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

  <implementation-priorities>
    <prestart must="true">
      <item>Schemas: create strict v1 stubs for all referenced schemas with $schema, $id (URN form), version, and additionalProperties=false.</item>
      <item>Stubs: register all new tools; unimplemented tools must return {success:false,error:{code:"NOT_IMPLEMENTED"}} deterministically.</item>
      <item>Manifest: write run_manifest.json with run_id, utc_iso, global_seed, inputs_hash (sha256 of canonicalized inputs), provenance, tools_used[].</item>
      <item>Gates: add config/gates.web.yaml and config/gates.native.yaml matching caps/tolerances; enforce via validators.gates_check.</item>
      <item>Non-regression: implement validators.non_regression per tolerances and priorities; deterministic rationale string.</item>
      <item>Human Feedback CLI: ship human_feedback/feedback_cli.py (Typer) with schema validation; write runs/latest/human_feedback.json.</item>
      <item>Capture robustness: godot.capture_views always writes captures_meta.json (modes_realized, poses, seed) and degrades gracefully.</item>
      <item>Make/Tests: add smoke script and schema tests; ensure godot_missing path returns structured error without crash.</item>
    </prestart>
    <standards>
      <schema_ids>Use URNs: urn:starsystem:schema:&lt;name&gt;:v1</schema_ids>
      <seed_and_hash>Centralize seed cascade and inputs_hash in a shared utility; all tools echo seed_used.</seed_and_hash>
      <deterministic_ids>Use UUIDv5 for issues/replays; blake3 content-address for artifacts in provenance.</deterministic_ids>
      <safety_default>Default visibility is private unless validators.safety_tag.ok === true (explicit).</safety_default>
      <building_system_loader>builders.load_system validates and persists JSON spec without requiring Godot.</building_system_loader>
    </standards>
  </implementation-priorities>

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
    <item>Schemas frozen with $id/$schema/version; validation tests pass.</item>
    <item>validators.gates_check and validators.non_regression enforce rules deterministically with machine-readable rationale.</item>
    <item>Human feedback round produces a valid human_feedback.json saved under runs/latest.</item>
    <item>Capture pipeline reports modes_realized and never crashes on missing modes.</item>
    <item>All tools echo seed_used; artifacts consolidated under runs/latest; README updated with client hookup.</item>
  </definition-of-done>

  <response-style>
    <format>PLAN → ACTIONS → VERIFY → RESULTS → NEXT</format>
    <verbosity>terse, command-first, no hidden steps</verbosity>
    <disallowed>cloud calls, non-MCP I/O, interactive prompts without APPROVE_INSTALL=1</disallowed>
  </response-style>
</high-order-prompt>
```


