"""
Summary: CLI to print assembled prompts for Agent A/B/C by combining high-order and low-order templates.

Usage examples:
  python -m agents.emit_prompts --agent a
  python -m agents.emit_prompts --agent c --vars /home/colling/PCC-LanguageV2/agents/prompt_vars.json
"""

from __future__ import annotations

import argparse
from agents.prompting.prompt_engine import main as engine_main


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    # Delegate all parsing/validation to the engine
    return engine_main()


if __name__ == "__main__":
    raise SystemExit(main())


