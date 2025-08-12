"""
Summary: Prompt assembly utilities for high-order/low-order alignment across agents A/B/C.

This module loads the canonical high-order benchmark prompt and merges it with
role-specific low-order prompts to produce aligned, ready-to-paste prompts for
terminal-based Claude Code sessions. No API calls are made here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "agents" / "prompting"
LOW_ORDER_DIR = PROMPTS_DIR / "low_order"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Missing prompt file: {path}")


def _render_template(template_str: str, variables: Dict[str, Any]) -> str:
    rendered = template_str
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered


def load_high_order_prompt() -> str:
    return _read_text(PROMPTS_DIR / "high_order_benchmark.md")


def load_low_order_prompt(agent_key: str) -> str:
    key = agent_key.strip().lower()
    name_map = {"a": "agent_a.md", "b": "agent_b.md", "c": "agent_c.md"}
    if key not in name_map:
        raise SystemExit("agent_key must be one of: a, b, c")
    return _read_text(LOW_ORDER_DIR / name_map[key])


def assemble_prompt(agent_key: str, variables: Dict[str, Any]) -> str:
    high = load_high_order_prompt()
    low = load_low_order_prompt(agent_key)

    high_r = _render_template(high, variables)
    low_r = _render_template(low, variables)

    # Clear separators that are friendly to terminal copy/paste
    sections = [
        "=== HIGH-ORDER BENCHMARK (source of truth) ===\n",
        high_r.strip() + "\n\n",
        "=== LOW-ORDER TASK (must align with the benchmark) ===\n",
        low_r.strip() + "\n",
    ]
    return "".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a fully assembled prompt for an agent (A/B/C) by merging the "
            "high-order benchmark with the agent's low-order instructions."
        )
    )
    parser.add_argument("--agent", "-a", required=True, choices=["a", "b", "c"], help="Agent key: a, b, or c")
    parser.add_argument(
        "--vars",
        "-v",
        type=Path,
        help="Optional JSON file with template variables to substitute into prompts",
    )
    args = parser.parse_args(argv)

    variables: Dict[str, Any] = {}
    if args.vars:
        try:
            variables = json.loads(args.vars.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to load vars JSON: {exc}")

    # Provide some helpful defaults if not supplied
    variables.setdefault("repo_root", str(REPO_ROOT))
    variables.setdefault("project_name", "PCC-Language V2")

    prompt = assemble_prompt(args.agent, variables)
    sys.stdout.write(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


