Summary: Three-agent plan using terminal-based model sessions with high/low-order prompts (A=Claude 3 Haiku, B=GPT-4o, C=Claude Sonnet 4). No cloud keys in code; local-first execution.

Overview
- Orchestration happens in your terminal client (Claude Code), not inside this repo. This repo only provides deterministic tools, schemas, and prompts.
- Agents align via a canonical benchmark prompt (`agents/prompting/high_order_benchmark.md`) and role-specific low-order prompts under `agents/prompting/low_order/`.
- Model assignments:
  - Agent A (Generator): Claude 3 Haiku — fast drafting and minimal edits.
  - Agent B (Tester): GPT-4o — rigorous validation and surgical fixes.
  - Agent C (Supervisor): Claude Sonnet 4 — objective setting and gating.

Non-negotiable constraints (repeat every run)
- Alignment: Low-order prompts must obey the benchmark prompt; escalate on conflict.
- Determinism: Every randomized path accepts a seed and documents its cascade.
- Local-first: No API keys in code. Terminal sessions may use cloud models, but the repo remains keyless and runnable offline with stubs.
- Contracts: Tool I/O via strict JSON schemas. Artifacts under `./runs/` with manifests.

WEEK 1 — Prompt system + MCP stubs + Godot discovery (terminal agents only)
Goals
- Ship reusable high/low-order prompts and a CLI to assemble them for terminal use.
- Stand up a minimal MCP-only server skeleton with deterministic stubs (local-only, no keys).
- Godot 4 headless discovery with stubbed behavior when absent.

Deliverables
- Prompts: `agents/prompting/high_order_benchmark.md`, `agents/prompting/low_order/agent_{a,b,c}.md`.
- CLI: `python3 -m agents.emit_prompts --agent {a|b|c} [--vars vars.json]` writes terminal-ready prompts.
- Agent mapping: `agents/agent_config.yaml` (A=Claude 3 Haiku, B=GPT-4o, C=Claude Sonnet 4).
- MCP skeleton and tools mirrors prior plan (local-only, deterministic, stubs OK when Godot missing).

Definition of Done (W1)
- Prompts render cleanly; low-order copies explicitly reference alignment to the benchmark.
- MCP server runs with tool registration; stub mode passes tests when Godot is absent.

WEEK 2 — Schemas locked, gates enforced, human feedback CLI (agents supervise via terminal)
Goals
- Freeze schemas with `$id`, `additionalProperties:false`. Enforce acceptance gates.
- Add human feedback CLI and deterministic capture/meta artifacts.
- Maintain terminal-based supervision: Agent C issues objectives and acceptance lists; Agent B validates; Agent A applies minimal edits.

Deliverables
- Schemas under `schemas/` frozen and versioned.
- Gates enforcement tools under `validators.*` (MCP contract unchanged).
- `human_feedback/feedback_cli.py` with validated outputs saved to the latest run.

Definition of Done (W2)
- Gate violations are deterministic and machine-verifiable.
- A human feedback round produces valid `human_feedback.json`.

WEEK 3 — Sequential loop orchestration (local tools) with terminal agents
Goals
- Wire a deterministic 3-stage loop (Worldsmith → Vision-QA → Fixer) using only local MCP tools.
- Agents run in terminal tabs using assembled prompts; repo stays keyless.
- Optional local VLM via Ollama; otherwise stats-only QA path.

Deliverables
- `crew/run_sequential.py`, `crew/*` deterministic policies; no remote calls.
- `scripts/run_loop.sh` to start MCP and run the loop with budgets/timeouts from `.env.example`.

Definition of Done (W3)
- End-to-end loop executes ≤3 fixer iterations, produces artifacts under `./runs/latest`.
- README documents how to point Claude Code to the MCP server, and how to use prompt assembly with A/B/C model tabs.

Agent operating rules (terminal)
- Agent C (Claude Sonnet 4): sets a tight objective and acceptance checklist per iteration; rejects scope creep.
- Agent B (GPT-4o): executes repros, reports crisp findings, and proposes smallest fixes; never broad rewrites.
- Agent A (Claude 3 Haiku): applies the smallest viable edits and exposes seeds; writes concise rationales.


