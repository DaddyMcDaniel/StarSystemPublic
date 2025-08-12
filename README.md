# StarSystem + The Forge (Pre-alpha)
Local-first, MCP-driven sandbox hub and creator tool. Week-1 scaffold.

## Quick Start

```bash
# Install dependencies
make install

# Test 3-agent handshake
make agents

# Run smoke test pipeline
make smoke

# Start MCP server (stdio mode)
make mcp-server
```

## Terminal agent prompts (high/low-order)

- Use the CLI to emit terminal-ready prompts assembled from a canonical benchmark plus role-specific low-order instructions:

```bash
python3 -m agents.emit_prompts --agent a > agents/memory/agent_a_prompt.txt
python3 -m agents.emit_prompts --agent b > agents/memory/agent_b_prompt.txt
python3 -m agents.emit_prompts --agent c > agents/memory/agent_c_prompt.txt
```

- Model assignments for terminal sessions:
  - Agent A: Claude 3 Haiku
  - Agent B: GPT-4o-mini (GPT-5 nano fallback)
  - Agent C: Claude Sonnet 4

## Claude Code Client Configuration

### stdio Transport (Recommended)
```bash
python3 mcp_server/server.py
```

### WebSocket Transport (Alternative)
```bash
# Start server on port 5174
TRANSPORT=ws python3 mcp_server/server.py
```

## Godot Discovery Order

The system discovers Godot binaries in this order:
1. `godot4-headless`
2. `godot4`  
3. `godot`
4. `./godot.AppImage --headless`

Discovery results are cached in `.cache/godot_path`.

## Architecture

- **MCP Server**: `mcp_server/server.py` - Headless Godot utilities and generators
- **3-Agent System**: A(generate) → B(test) → C(validate) with deterministic handshakes
- **Smoke Pipeline**: `generate → apply_patch → test_headless → capture_views → dump_scene`
- **Schemas**: Strict input/output validation in `schemas/`
- **Artifacts**: Deterministic runs in `runs/` with atomic `latest` symlinks

## Week-1 Goals

✅ **MCP-only Python server** exposing headless Godot utilities and local generators/validators  
✅ **Repo scaffolding** with strict schemas, tests, and runs/ artifact discipline  
✅ **Minimal generators** (maze/planet) with nav feasibility checks (stubs ok)  
✅ **Smoke path** validates loop: generate → apply_patch → test_headless → capture_views → dump_scene  
✅ **Claude Code ↔ MCP** hookup documentation (stdio/WS) and local Godot discovery

The canonical high-order benchmark lives at `agents/prompting/high_order_benchmark.md`. Low-order prompts for each role live under `agents/prompting/low_order/` and must align to the benchmark.