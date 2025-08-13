Summary: Week-3 Low-Order Prompt (executor). Implements the alpha loop with OpenGL runtime default, miniplanet + FP nav, and Building System MVP. Includes optional changes with a FORGE_RUNTIME toggle.

```xml
<low-order-prompt version="3.0" role="executor">
  <context>
    <phase week="3" seed="${GLOBAL_SEED}" gates_profile="web"/>
    <repo root="${ABS_PATH_TO_REPO}">
      <branch>main</branch>
      <status>clean|dirty</status>
      <name>starsystem</name>
    </repo>
    <contracts_source>agents/prompting/high_order_benchmark_week3.md</contracts_source>
  </context>

  <objective>
    Ship alpha proof: generate and run a mini-planet seed with 3D first-person navigation using gl viewer; deliver Building MVP (grid/snap/undo) and optional Power + replay + HUD + publish gate per HOP priorities.
  </objective>

  <runtime>
    <toggle env="FORGE_RUNTIME">gl (default) | godot</toggle>
    <policy>If godot and missing binaries: MCP returns deterministic stubs; do not block gl path.</policy>
  </runtime>

  <priorities>
    <p1>gl viewer runs mini-planet + FP nav</p1>
    <p2>Building MVP (grid/snap/undo ledger)</p2>
    <p3>Optional: Power network + overlay</p3>
    <p4>Optional: Deterministic replay + microsim preview</p4>
    <p5>Optional: HUD + Performance Pass + publish gate</p5>
  </priorities>

  <plan>
    <steps>
      <step id="S1">gl runtime: implement renderer/pcc_simple_viewer.py with WASD+mouse-look, seed load, and loop entry; add scripts/run_gl.py.</step>
      <step id="S2">Dual-world hot-swap: forge/core/engine.py (apply_oplog_to_sim, tick boundary), forge/core/oplog.py, MCP tool world.hot_swap.</step>
      <step id="S3">Building MVP: forge/modules/build/grid.py (chunked grid), forge/modules/validators/placement.py, forge/modules/undo/ledger_writer.py; MCP builders.validate_placement/apply_placement and ledger.write.</step>
      <step id="S4">Power network (optional): forge/modules/power/graph.py, overlay.py; MCP power.solve; reuse schematic_card I/O ports.</step>
      <step id="S5">Replay + microsim (optional): scripts/replay_cli.py, forge/core/preview/microsim.py; MCP preview.microsim; write replays under runs/{run_id}/replays/.</step>
      <step id="S6">HUD + perf pass (optional): forge/modules/ui/hud.py, scripts/perf_pass.py; validators.performance_pass; gate configs reused.</step>
      <step id="S7">Publish dry run (optional): scripts/publish_dry_run.py; checks safety_tag + performance_pass green.</step>
      <step id="S8">Layer UX (optional): forge/modules/ui/mode_switcher.py; MCP world.list_layers/world.edit_layer for orbit/surface/subsurface.</step>
    </steps>
    <safety>
      <policy>Restrict mutation to forge/**, renderer/**, scripts/**, examples/projects/**; no arbitrary exec.</policy>
      <errors>Unimplemented paths return {success:false,error:{code:"NOT_IMPLEMENTED"}}; never crash.</errors>
    </safety>
    <determinism>
      <seed_cascade>global_seed → world_seed → tool_seed → chip_seed</seed_cascade>
      <replay>fixed-step sim; replays must match bit-for-bit on repeat</replay>
    </determinism>
  </plan>

  <commands shell="bash">
    <![CDATA[
set -euo pipefail

# Ensure config toggle exists
grep -q '^FORGE_RUNTIME=' .env 2>/dev/null || echo 'FORGE_RUNTIME=gl' >> .env || true

# gl quick run helper
cat > scripts/run_gl.py <<'EOF'
import os, sys, subprocess
os.environ.setdefault('FORGE_RUNTIME','gl')
subprocess.run([sys.executable, 'renderer/pcc_simple_viewer.py'] + sys.argv[1:], check=False)
EOF

echo "Week-3 executor plan scaffold completed. Implement modules per plan."
    ]]>
  </commands>

  <verify>
    <acceptance>
      <check>python3 scripts/run_gl.py runs the mini-planet with FP nav.</check>
      <check>Place/move/delete 500+ blocks; undo/redo persists across save/load.</check>
      <check>Optional features produce artifacts and pass gates if enabled.</check>
    </acceptance>
  </verify>

  <results>
    <summary>Alpha loop online with gl default; Building MVP ready; optional subsystems gated but integrated.</summary>
  </results>
</low-order-prompt>
```


