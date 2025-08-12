import os, json, time, hashlib, pathlib, sys
from datetime import datetime, timezone

RUNS = pathlib.Path("runs")
RUNS.mkdir(parents=True, exist_ok=True)

def seed_cascade(tool, seed, inputs):
    frozen = json.dumps(inputs, sort_keys=True)
    h = hashlib.sha256((tool + str(seed) + frozen).encode()).hexdigest()
    return int(h[:12], 16)  # small int

def log_line(tool, payload):
    outdir = RUNS / "latest"
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(outdir / "logs" / "agents.jsonl", "a") as f:
        f.write(json.dumps({"ts": ts, "tool": tool, **payload}) + "\n")

def provider_available(name):
    if name == "gpt5_nano":
        return bool(os.getenv("OPENAI_API_KEY"))
    if name.startswith("claude"):
        return bool(os.getenv("CLAUDE_CODE_AVAILABLE", "1"))
    return False

def call_agent(name, role, message, seed):
    cfg_ok = provider_available(name)
    if not cfg_ok:
        payload = {"success": False, "error": {"code": "agent_unavailable", "message": f"{name} not configured"}, "role": role, "message": message}
        log_line(f"agent_{role}", payload)
        return payload
    sid = seed_cascade(f"agent_{role}", seed, {"m": message})
    reply = f"[{role}:{name}] seed={sid} :: {message[:120]}"
    payload = {"success": True, "role": role, "reply": reply, "seed_used": sid}
    log_line(f"agent_{role}", payload)
    return payload