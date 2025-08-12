#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def cleanup_supertux_memory(retain_top_k_skills: int = 50, retain_tips_min_count: int = 2) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    mem_file = root / "agents" / "memory" / "agent_b_bb_supertux_memory.json"
    if not mem_file.exists():
        return {"status": "missing"}
    data = json.loads(mem_file.read_text())
    # keep skills with highest score*uses
    skills = data.get("skills", [])
    skills.sort(key=lambda s: float(s.get("score", 0.0)) * (1.0 + 0.2 * int(s.get("uses", 0))), reverse=True)
    data["skills"] = skills[: max(1, int(retain_top_k_skills))]
    # prune weak tips
    tips = data.get("tips_seen", {})
    data["tips_seen"] = {k: v for k, v in tips.items() if int(v) >= int(retain_tips_min_count)}
    mem_file.write_text(json.dumps(data, indent=2))
    return {"status": "ok", "skills": len(data["skills"]), "tips": len(data["tips_seen"]) }


if __name__ == "__main__":
    out = cleanup_supertux_memory()
    print(json.dumps(out, indent=2))


