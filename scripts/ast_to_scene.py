#!/usr/bin/env python3
"""
CLI: Convert PCC AST (.pcc) to a grounded scene JSON for the viewer.
Usage:
  python scripts/ast_to_scene.py tests/examples/game_*.pcc -o runs/<name>_scene.json
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forge.modules.worldgen_from_ast import generate_scene_from_pcc


def main():
    ap = argparse.ArgumentParser(description="Convert PCC AST to grounded scene JSON")
    ap.add_argument("pcc_file", help="Path to .pcc AST file")
    ap.add_argument("-o", "--out", help="Output scene JSON path", default=None)
    args = ap.parse_args()

    pcc_path = Path(args.pcc_file)
    if not pcc_path.exists():
        print(f"❌ PCC file not found: {pcc_path}")
        return 1

    ast = json.loads(pcc_path.read_text())
    scene = generate_scene_from_pcc(ast, seed=ast.get("planet_seed"))

    out_path = Path(args.out) if args.out else Path("runs") / (pcc_path.stem + "_scene.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene, indent=2))
    print(f"✅ Wrote scene: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


