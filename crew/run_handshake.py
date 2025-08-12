import json, pathlib, time
from agents_local import call_agent, RUNS
from datetime import datetime, timezone

def ensure_run():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"{stamp}-seed0"
    out.mkdir(parents=True, exist_ok=True)
    latest = RUNS / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        import shutil
        shutil.rmtree(latest)
    latest.symlink_to(out.name, target_is_directory=True)
    (out / "logs").mkdir(exist_ok=True)
    return out

def main():
    out = ensure_run()
    seed = 0
    a = call_agent("claude_haiku", "A", "ping: blueprint for minimal scene + MCP tool list", seed)
    b_msg = ("transform: echo A plan into code tasks" if a.get("success") else "diagnose: missing A")
    b = call_agent("gpt5_nano", "B", b_msg, seed)
    c_msg = ("validate: ask human to confirm visuals; fallback stats-only" if b.get("success") else "diagnose: missing B")
    c = call_agent("claude_sonnet_4", "C", c_msg, seed)

    result = {"success": all([a.get("success"), b.get("success"), c.get("success")]),
              "comms": {"A": a.get("success"), "B": b.get("success"), "C": c.get("success")},
              "paths": {"run_dir": str(out)}}
    with open(out / "handshake_result.json", "w") as f:
      json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()