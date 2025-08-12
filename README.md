# StarSystem + The Forge (Pre-alpha)
Local-first, MCP-driven sandbox hub and creator tool. Week-1 scaffold.

## Terminal agent prompts (high/low-order)

- Use the CLI to emit terminal-ready prompts assembled from a canonical benchmark plus role-specific low-order instructions:

```bash
python3 -m agents.emit_prompts --agent a > agents/memory/agent_a_prompt.txt
python3 -m agents.emit_prompts --agent b > agents/memory/agent_b_prompt.txt
python3 -m agents.emit_prompts --agent c > agents/memory/agent_c_prompt.txt
```

- Model assignments for terminal sessions:
  - Agent A: Claude 3 Haiku
  - Agent B: GPT-4o
  - Agent C: Claude Sonnet 4

The canonical high-order benchmark lives at `agents/prompting/high_order_benchmark.md`. Low-order prompts for each role live under `agents/prompting/low_order/` and must align to the benchmark.
MD < /dev/null
