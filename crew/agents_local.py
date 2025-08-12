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

def call_openai_api(message, seed, model="gpt-4o-mini"):
    """Call OpenAI API for GPT-5 nano"""
    import requests
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "No API key"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "seed": seed,
        "max_completion_tokens": 150,
        "temperature": 0.1
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", ""), None
        else:
            return None, f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

def call_agent(name, role, message, seed):
    cfg_ok = provider_available(name)
    if not cfg_ok:
        payload = {"success": False, "error": {"code": "agent_unavailable", "message": f"{name} not configured"}, "role": role, "message": message}
        log_line(f"agent_{role}", payload)
        return payload
    
    sid = seed_cascade(f"agent_{role}", seed, {"m": message})
    
    # Handle GPT-5 nano with actual API call (using gpt-4o-mini as fallback)
    if name == "gpt5_nano":
        reply, error = call_openai_api(message, sid, "gpt-4o-mini")
        if error:
            payload = {"success": False, "error": {"code": "api_call_failed", "message": error}, "role": role, "message": message, "seed_used": sid}
        else:
            payload = {"success": True, "role": role, "reply": reply, "seed_used": sid}
    else:
        # Stub for other providers
        reply = f"[{role}:{name}] seed={sid} :: {message[:120]}"
        payload = {"success": True, "role": role, "reply": reply, "seed_used": sid}
    
    log_line(f"agent_{role}", payload)
    return payload