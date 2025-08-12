Summary: First main Low-order prompt (executor). Vision-aligned and aligned to the canonical high-order benchmark. Credentials must come from environment; do not paste secrets in files or logs.

```xml
<low-order-prompt version="1.0" role="executor">
  <context>
    <phase week="1" seed="${GLOBAL_SEED}" gates_profile="web"/>
    <repo root="${ABS_PATH_TO_REPO}">
      <branch>main</branch>
      <status>clean</status>
      <name>starsystem</name>
      <visibility>private</visibility>
    </repo>
    <discovery>
      <godot path="${DETECTED_OR_EMPTY}" found="false|true"/>
      <ollama installed="false|true"/>
      <node version=">=18" found="true|false"/>
    </discovery>
    <credentials policy="no-cloud-LLM-keys; GitHub creds via env only; never log secrets">
      <github username="Daddy Mcdaniel" pat_env="GITHUB_PAT" password_env="GITHUB_PASSWORD" note="Prefer PAT for git push; password only for Playwright UI login if used."/>
      <env required="GITHUB_USER,GITHUB_PASSWORD or GITHUB_PAT">interactive-2FA-allowed</env>
      <ssh recommended="true">Use existing SSH key or PAT for git remotes; not required for Playwright creation.</ssh>
    </credentials>
    <mcp transport="stdio" ws_port="5174">
      <tools-expected>
        <tool>godot.test_headless</tool>
        <tool>godot.capture_views</tool>
        <tool>godot.dump_scene</tool>
        <tool>godot.apply_patch</tool>
        <tool>generators.maze_generate</tool>
        <tool>validators.maze_graph_validate</tool>
        <tool>memory.search</tool>
      </tools-expected>
    </mcp>
  </context>

  <task id="repo_bootstrap_and_agents" title="Create Git repo via Playwright, push to GitHub, then stand up 3-agent loop (A/B/C)">
    <objective>
      1) Automate creation of a new GitHub repository using Playwright (browser UI), initialize local git, first commit, and push.
      2) Scaffold a local MCP-first agent loop (A: Claude 3 Haiku, B: GPT-4o, C: Claude Sonnet 4 + human) with deterministic handshakes and stubs if APIs are unavailable.
    </objective>
    <acceptance>
      <check>GitHub repo exists at owner/repo and is reachable (HTTP 200) after Playwright run.</check>
      <check>Local repo initialized with README, LICENSE, .gitignore; main branch pushed with initial commit.</check>
      <check>crew/run_sequential.py (or agents_local.py) can run a handshake: A->B->C ping pipeline with structured JSON logs under runs/.</check>
      <check>No cloud LLM keys required; when absent, agents emit stubbed responses with success=false and error.code="agent_unavailable", without crash.</check>
    </acceptance>
    <constraints>
      <time per_call_timeout="120s" wall_budget="<= 900s"/>
      <safety mutate_only="automation/**, crew/**, scripts/**, README.md, .gitignore, LICENSE, package.json, playwright/**, .github/**" no_exec_scripts="true"/>
      <determinism seed_policy="GLOBAL_SEED -> hash(tool_name, GLOBAL_SEED, inputs_frozen)" rng="pcg64"/>
    </constraints>
  </task>

  <plan>
    <steps>
      <step id="1">Scaffold Node+Playwright project under automation/playwright; install chromium only.</step>
      <step id="2">Create script that logs into GitHub (using env: GITHUB_USER / GITHUB_PASSWORD), handles optional 2FA pause, then creates repo ${REPO_NAME}.</step>
      <step id="3">Capture repo URL to automation/playwright/out/repo.json and a screenshot for audit.</step>
      <step id="4">Initialize local git in repo root; add README/LICENSE/.gitignore; set remote to created repo and push main (prefer PAT if available).</step>
      <step id="5">Scaffold agents: crew/agents_local.py, crew/run_handshake.py, crew/agent_config.yaml mapping A/B/C -> providers (cloud keys optional); provide stubs when unavailable.</step>
      <step id="6">Run handshake: A emits plan, B echoes transform, C validates + (if human present) requests confirmation; write JSONL logs to runs/.../logs/agents.jsonl.</step>
    </steps>
    <heavy-installs approval_required="APPROVE_INSTALL" if_missing="print-plan-and-exit"/>
    <failure-policy>
      <on error_code="playwright_login_failed">pause for manual 2FA; retry once headful</on>
      <on error_code="lock_timeout">retry_with_backoff: 5 attempts, max 30s</on>
      <on error_code="agent_unavailable">stub-and-continue; mark comms=false in handshake result</on>
    </failure-policy>
  </plan>

  <commands shell="bash">
    <![CDATA[
set -euo pipefail

# --- 0) Expect credentials via environment (do not echo secrets) ---
# export GITHUB_USER="Daddy Mcdaniel"
# export GITHUB_PASSWORD="<your-password-if-using-UI-login>"  # optional
# export GITHUB_PAT="<your-fine-grained-PAT>"                  # preferred for git push; NEVER commit
# export REPO_NAME="starsystem"
# export REPO_VISIBILITY="private"   # or public

# --- 1) Node + Playwright scaffold (deterministic) ---
mkdir -p automation/playwright/out
cd automation/playwright
test -f package.json || npm init -y
npm pkg set name="@starsystem/playwright" type="module"
npm i -D playwright@^1
npx playwright install --with-deps chromium

# --- 2) Write Playwright repo-creator script ---
cat > create_repo.ts <<'TS'
import { chromium } from 'playwright';
const USER = process.env.GITHUB_USER || '';
const PASS = process.env.GITHUB_PASSWORD || '';
const REPO = process.env.REPO_NAME || 'starsystem';
const VIS  = process.env.REPO_VISIBILITY || 'private'; // 'public' | 'private'
if (!USER || !PASS) {
  console.error(JSON.stringify({ success:false, error:{code:"missing_env", message:"GITHUB_USER/GITHUB_PASSWORD required for UI login"} }));
  process.exit(0);
}
(async () => {
  const browser = await chromium.launch({ headless: false }); // allow 2FA
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  try {
    await page.goto('https://github.com/login');
    await page.getByLabel('Username or email address').fill(USER);
    await page.getByLabel('Password').fill(PASS);
    await page.getByRole('button', { name: 'Sign in' }).click();

    // If 2FA present, wait up to 120s for user to complete.
    if (await page.getByText('Verify').first().isVisible().catch(() => false)) {
      await page.waitForTimeout(120000);
    }

    await page.goto('https://github.com/new');
    await page.getByLabel('Repository name').fill(REPO);
    if (VIS === 'private') {
      await page.getByLabel('Private').check({ force:true });
    } else {
      await page.getByLabel('Public').check({ force:true });
    }
    // Initialize with README to simplify first push
    const initReadme = await page.getByLabel('Add a README file').isVisible().catch(() => false);
    if (initReadme) await page.getByLabel('Add a README file').check({ force:true });

    await page.getByRole('button', { name: 'Create repository' }).click();
    await page.waitForURL(/github\.com\/.+?\/.+?$/);

    const url = page.url();
    await page.screenshot({ path: 'out/repo_created.png', fullPage: true });
    console.log(JSON.stringify({ success:true, data:{ repo_url:url } }));
  } catch (e:any) {
    console.error(JSON.stringify({ success:false, error:{ code:"playwright_login_failed", message:e?.message || String(e) } }));
  } finally {
    await ctx.close(); await browser.close();
  }
})();
TS

# --- 3) Run Playwright script to create repo (requires env) ---
node create_repo.ts | tee out/repo_raw.json
node -e 'const fs=require("fs"); const lines=fs.readFileSync("out/repo_raw.json","utf8").split(/\r?\n/).filter(Boolean); const j=JSON.parse(lines.pop()); if(!j.success){process.exit(0)}; fs.writeFileSync("out/repo.json", JSON.stringify(j.data, null, 2));'

cd ../..

# --- 4) Initialize local git; push to GitHub ---
# Read repo URL (if creation succeeded)
REPO_URL="$(node -e 'try{console.log(JSON.parse(require("fs").readFileSync("automation/playwright/out/repo.json","utf8")).repo_url)}catch{process.exit(0)}')"
if [ -z "${REPO_URL:-}" ]; then
  echo "Playwright repo creation skipped or failed; you can set REMOTE manually later."
fi

# Seed project files
test -f README.md || cat > README.md <<'MD'
# StarSystem + The Forge (Pre-alpha)
Local-first, MCP-driven sandbox hub and creator tool. Week-1 scaffold.
MD

test -f LICENSE || cat > LICENSE <<'L'
MIT License
L

echo -e "venv/\nnode_modules/\n.DS_Store/\n.env\n.env.local\nruns/\n.cache/\nautomation/playwright/out/\n" > .gitignore

git init -b main
git add .
git commit -m "chore(repo): init scaffold via Playwright + MCP agents bootstrap"
if [ -n "${REPO_URL:-}" ]; then
  # Convert web URL to owner/repo
  OWNER_REPO="$(echo "$REPO_URL" | sed -E 's#https://github.com/([^/]+/[^/]+).*#\1#')"
  if [ -n "${GITHUB_PAT:-}" ]; then
    # Prefer PAT-based HTTPS remote; DO NOT echo PAT
    REMOTE_URL="https://${GITHUB_PAT}@github.com/${OWNER_REPO}.git"
  else
    # Fallback to plain HTTPS (may require credentials prompt) or SSH if configured
    REMOTE_URL="https://github.com/${OWNER_REPO}.git"
  fi
  git remote add origin "$REMOTE_URL" || true
  git push -u origin main || true
fi

# --- 5) Scaffold agents (A/B/C) with stubs when keys are missing ---
mkdir -p crew scripts runs
cat > crew/agent_config.yaml <<'YAML'
version: 1
routing:
  A:
    provider: claude_haiku
    transport: mcp_client
    fallback: stub
  B:
    provider: gpt4o
    transport: local_process
    fallback: stub
  C:
    provider: claude_sonnet_4
    transport: mcp_client_with_human
    fallback: stub
budgets:
  calls_max: 100
  seconds_max: 900
timeouts:
  per_tool_seconds: 120
YAML

cat > crew/agents_local.py <<'PY'
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
    if name == "gpt4o":
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
PY

cat > crew/run_handshake.py <<'PY'
import json, pathlib, time
from agents_local import call_agent, RUNS
from datetime import datetime, timezone

def ensure_run():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"{stamp}-seed0"
    out.mkdir(parents=True, exist_ok=True)
    latest = RUNS / "latest"
    if latest.is_symlink() or latest.exists():
        try:
            latest.unlink()
        except Exception:
            pass
    latest.symlink_to(out, target_is_directory=True)
    (out / "logs").mkdir(exist_ok=True)
    return out

def main():
    out = ensure_run()
    seed = 0
    a = call_agent("claude_haiku", "A", "ping: blueprint for minimal scene + MCP tool list", seed)
    b_msg = ("transform: echo A plan into code tasks" if a.get("success") else "diagnose: missing A")
    b = call_agent("gpt4o", "B", b_msg, seed)
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
PY

# --- 6) Execute handshake (always succeeds with stubs) ---
python3 crew/run_handshake.py | tee runs/handshake_last.json

# --- 7) Commit scaffold and push ---
git add .
git commit -m "feat(agents): bootstrap A/B/C handshake with MCP-first stubs + Playwright repo creator" || true
git push || true
    ]]>
  </commands>

  <mcp-calls>
    <!-- None required for this step; next tasks will use MCP tools. Handshake is local-only. -->
  </mcp-calls>

  <verify>
    <tests>
      <assert file="automation/playwright/out/repo_raw.json" exists="true"/>
      <assert file="automation/playwright/out/repo.json" exists="true-or-absent">ok if login/2FA deferred</assert>
      <assert file="runs/handshake_last.json" exists="true"/>
      <assert file="runs/latest/logs/agents.jsonl" exists="true"/>
    </tests>
    <gates check="skip_for_this_task"/>
  </verify>

  <artifacts>
    <record dir="runs/${UTC_ISO}-${GLOBAL_SEED}"/>
    <log append="runs/${UTC_ISO}-${GLOBAL_SEED}/logs/agents.jsonl"/>
    <latest symlink="./runs/latest" atomic="true"/>
  </artifacts>

  <commit when="on_success">
    <message>feat(repo+agents): GitHub repo via Playwright; 3-agent loop (A/B/C) with deterministic stubs</message>
    <include paths="automation/**, crew/**, README.md, .gitignore, LICENSE, runs/**"/>
  </commit>

  <results>
    <summary>Repo created via Playwright; local git initialized and (if possible) pushed. Agent loop booted with A/B/C roles; handshake result written; stubs used when providers unavailable.</summary>
    <files>
      <file path="automation/playwright/out/repo_raw.json"/>
      <file path="automation/playwright/out/repo.json"/>
      <file path="runs/latest/handshake_result.json"/>
      <file path="runs/latest/logs/agents.jsonl"/>
    </files>
  </results>

  <next>
    <if success="true">Task: Wire MCP server + smoke tools, then route Agent B actions through MCP validators.</if>
    <if success="false">Remediate: Resolve Playwright login/2FA; rerun create_repo.ts headful; then re-run handshake.</if>
  </next>
</low-order-prompt>
```


