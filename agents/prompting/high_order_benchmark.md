Summary: Vision-aligned canonical high-order benchmark prompt (governor). All low-order prompts and actions must obey this.

```xml
<high-order-prompt version="1.0" role="governor">
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

  <observability>
    <logs tools_jsonl="runs/.../logs/tools.jsonl"/>
    <manifest file="runs/.../run_manifest.json"/>
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
</high-order-prompt>
```


