Summary: Week-2 Low-Order Prompt (executor). Scaffold, prioritize, and safely implement all changes proposed in Week-2 HOP.

```xml
<low-order-prompt version="2.0" role="executor">
  <context>
    <phase week="2" seed="${GLOBAL_SEED}" gates_profile="web"/>
    <repo root="${ABS_PATH_TO_REPO}">
      <branch>main</branch>
      <status>clean|dirty</status>
      <name>starsystem</name>
    </repo>
    <contracts_source>agents/prompting/high_order_benchmark_week2.md</contracts_source>
  </context>

  <objective>
    Deliver Week-2 readiness: frozen schemas, registered tools with deterministic stubs, gates/non-regression validators, human feedback CLI, capture robustness, and run_manifest fields.
  </objective>

  <plan>
    <steps>
      <step id="S1">Create v1 schema stubs with strict typing and URN $id (urn:starsystem:schema:&lt;name&gt;:v1). Files: blueprint_chip, replay, performance_pass, schematic_card, world_diff, provenance, safety_tag, ledger, building_system, patch_plan, gates, non_regression, human_feedback.</step>
      <step id="S2">Register new tools in MCP server; each unimplemented tool returns NOT_IMPLEMENTED with seed_used and execution_time_ms.</step>
      <step id="S3">Implement validators.gates_check reading config/gates.web.yaml and config/gates.native.yaml; output machine-readable violations and deltas.</step>
      <step id="S4">Implement validators.non_regression per HOP tolerances/priorities; emit deterministic rationale; enforce hard caps.</step>
      <step id="S5">Human Feedback CLI (Typer + jsonschema): validate rubric and issues; write runs/latest/human_feedback.json.</step>
      <step id="S6">Capture robustness: ensure captures_meta.json always includes modes_realized, camera poses, and seed; degrade without crash.</step>
      <step id="S7">Manifest: write run_manifest.json with required fields (run_id, utc_iso, global_seed, inputs_hash, provenance, tools_used[]).</step>
      <step id="S8">Make/Tests: add smoke.sh; schema tests; gates/non-regression tests; godot_missing stub tests.</step>
    </steps>
    <safety>
      <policy>Do not execute arbitrary scripts; restrict mutation to examples/projects/* and local schemas/config.</policy>
      <error_model>Unimplemented paths MUST return {success:false,error:{code:"NOT_IMPLEMENTED"}} deterministically.</error_model>
      <rollback>On gate_violation or failed non_regression, stop and write rationale; no destructive changes.</rollback>
    </safety>
    <determinism>
      <seed_cascade>global_seed → world_seed → tool_seed → chip_seed</seed_cascade>
      <hashing>inputs_hash = sha256(canonicalized_inputs)</hashing>
      <ids>uuidv5 for issues/replays; blake3 content-address for artifacts</ids>
    </determinism>
  </plan>

  <commands shell="bash">
    <![CDATA[
set -euo pipefail

# S1: Schema stubs (create if missing)
mkdir -p schemas config human_feedback tests scripts
for f in \
  blueprint_chip.v1.schema.json replay.v1.schema.json performance_pass.v1.schema.json \
  schematic_card.v1.schema.json world_diff.v1.schema.json provenance.v1.schema.json \
  safety_tag.v1.schema.json ledger.v1.schema.json building_system.v1.schema.json \
  patch_plan.v1.schema.json gates.v1.schema.json non_regression.v1.schema.json human_feedback.v1.schema.json; do
  if [ ! -f "schemas/$f" ]; then
    echo '{"$schema":"http://json-schema.org/draft-07/schema#","$id":"urn:starsystem:schema:REPLACE:v1","version":1,"type":"object","additionalProperties":false}' > "schemas/$f"
  fi
done

# S3: Gates configs (minimal placeholders)
cat > config/gates.web.yaml <<'EOF'
hard_caps:
  tri_count: 120000
  materials: 32
priorities: [readability, perf_proxy, tri_count]
tolerances:
  tri_count:
    worsen_max_pct: 2.0
  materials:
    worsen_max: 1
tie_breakers: [perf_proxy, tri_count]
nan_policy: treat_as_worst
EOF
cp -f config/gates.web.yaml config/gates.native.yaml

# S5: Human Feedback CLI placeholder (manual implementation step referenced in HOP)
mkdir -p human_feedback
if [ ! -f human_feedback/feedback_cli.py ]; then
  echo "# TODO: implement Typer CLI validating human_feedback.v1.schema.json" > human_feedback/feedback_cli.py
fi

echo "Week-2 scaffold steps completed (files written)."
    ]]>
  </commands>

  <verify>
    <checks>
      <check>Schemas exist under schemas/*.v1.schema.json and are well-formed JSON.</check>
      <check>config/gates.web.yaml present and valid YAML.</check>
      <check>human_feedback/feedback_cli.py present.</check>
    </checks>
  </verify>

  <results>
    <summary>Week-2 contracts scaffolded; ready for deterministic stubs and validators implementation.</summary>
  </results>
</low-order-prompt>
