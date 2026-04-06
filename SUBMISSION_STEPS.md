# SUBMISSION_STEPS.md — Hackathon Submission Guide

**Deadline: April 8, 2026, 11:59 PM IST**

Work through these steps in order. Each step has a verification command so you
know it worked before moving to the next one.

---

## Step 0 — Prerequisites (do once)

```bash
# 1. Verify Python 3.10+
python3 --version     # must be 3.10, 3.11, or 3.12

# 2. Verify uv is installed
uv --version
# If missing: curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Verify Docker is running
docker info           # must not error

# 4. Verify HuggingFace CLI is installed
huggingface-cli --version
# If missing: pip install huggingface_hub

# 5. Verify openenv-core CLI is available
openm --help
# If missing: pip install "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3"

# 6. Set your HF token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_...

# 7. Login to HuggingFace CLI
huggingface-cli login --token $HF_TOKEN
```

---

## Step 1 — Navigate to the project

```bash
cd /Users/gopalsaraf/Projects/recruitment-screening-env
```

All subsequent commands are run from this directory.

---

## Step 2 — Generate uv.lock (required by Dockerfile)

```bash
uv lock
```

✅ **Verify:** A `uv.lock` file appears in the directory.

```bash
ls -la uv.lock    # should exist and be non-empty
```

---

## Step 3 — Build the Docker image

```bash
docker build -t recruitment-screening-env .
```

This will take 3–8 minutes on first run (downloading base image, installing deps).
Subsequent builds use cache and are much faster.

✅ **Verify:**
```bash
docker images | grep recruitment-screening-env
# Should show: recruitment-screening-env   latest   <hash>   <size>
```

---

## Step 4 — Run the container locally

```bash
docker run -d -p 8000:8000 --name rse-test recruitment-screening-env
```

Wait ~10 seconds for the server to start, then:

✅ **Verify health:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"} or similar 200 response
```

✅ **Verify web UI** (optional — open in browser):
```
http://localhost:8000/web
```

✅ **Verify reset endpoint:**
```bash
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}' | python3 -m json.tool
```

**Cleanup test container:**
```bash
docker stop rse-test && docker rm rse-test
```

---

## Step 5 — Run openenv validate

```bash
openm validate
```

This checks:
- `openenv.yaml` is valid
- Dockerfile builds correctly
- All required endpoints respond
- Models are typed correctly

✅ **Verify:** All checks pass with no errors.

If it fails, the error message will tell you exactly what's wrong.

---

## Step 6 — Run the inference script end-to-end

This is the most important test. The hackathon evaluators run this exact script.

```bash
# Make sure HF_TOKEN is set
export HF_TOKEN=hf_...

# Run inference (auto-starts Docker, runs 3 tasks, stops Docker)
uv run inference.py
```

You should see structured JSON output on stdout like:
```json
{"event": "[START]", "task_id": "task_1_easy", "difficulty": "easy", "seed": 42, ...}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 1, "tool_name": "get_task", "reward": 0.0, ...}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 3, "tool_name": "submit_decision", "reward": 0.85, ...}
{"event": "[END]",   "task_id": "task_1_easy", "final_reward": 0.85, "total_steps": 3, "success": true, ...}
{"event": "[START]", "task_id": "task_2_medium", ...}
...
{"event": "[SUMMARY]", "tasks_run": 3, "average_reward": 0.73, ...}
```

✅ **Verify:**
- No Python errors or exceptions
- `[START]`, `[STEP]`, `[END]` events appear for all 3 tasks
- `[SUMMARY]` appears at the end
- `average_reward` is between 0.0 and 1.0 (not 0.0 for all tasks)
- Each `final_reward` in `[END]` is between 0.0 and 1.0

**If rewards are all 0.0:** The LLM isn't formatting the decision correctly.
Check `CLAUDE.md` section 11 for common issues.

**If Docker fails to start:** Run with `ENV_BASE_URL` to use an already-running container:
```bash
docker run -d -p 8000:8000 recruitment-screening-env
export ENV_BASE_URL=http://localhost:8000
uv run inference.py
```

---

## Step 7 — Push to HuggingFace Spaces

```bash
openm push <your-hf-username>/recruitment-screening-env
```

Replace `<your-hf-username>` with your actual HuggingFace username (e.g., `gopalsaraf`).

This command:
1. Packages your environment
2. Creates a HuggingFace Space repository
3. Uploads all files
4. Triggers a Docker build on HuggingFace's servers

The push takes 1–3 minutes. The Space build takes an additional 3–10 minutes.

✅ **Verify the Space is live:**

Go to: `https://huggingface.co/spaces/<your-hf-username>/recruitment-screening-env`

The Space should show "Running" status (green dot). Click the Space URL to see the web UI.

✅ **Verify the Space health endpoint:**
```bash
curl https://<your-hf-username>-recruitment-screening-env.hf.space/health
# Expected: 200 OK response
```

Note: HuggingFace Space URLs use hyphens and follow the pattern:
`https://<username>-<space-name>.hf.space`

---

## Step 8 — Run inference against the deployed Space (optional but recommended)

```bash
export HF_TOKEN=hf_...
export ENV_BASE_URL=https://<your-hf-username>-recruitment-screening-env.hf.space

uv run inference.py
```

This verifies the deployed Space behaves exactly like your local container.

---

## Step 9 — Submit on the Hackathon Dashboard

1. Go to: `https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard`
2. Scroll to the submission section
3. Paste your HuggingFace Space URL:
   ```
   https://huggingface.co/spaces/<your-hf-username>/recruitment-screening-env
   ```
4. Submit

✅ **Only the team leader can submit.**

---

## Step 10 — Re-submission (if needed)

Multiple submissions are allowed. **The most recent submission counts.**

If you make changes:
```bash
# 1. Make your code changes
# 2. Rebuild Docker
docker build -t recruitment-screening-env .

# 3. Re-run inference to verify
uv run inference.py

# 4. Re-push to HuggingFace
openm push <your-hf-username>/recruitment-screening-env

# 5. Re-submit the same URL on the dashboard (it auto-updates)
```

---

## Pre-Submission Checklist

Run through this before hitting Submit. The hackathon runs **automated checks** —
all 5 must pass or you risk disqualification.

- [ ] `docker build -t recruitment-screening-env .` completes without errors
- [ ] `curl http://localhost:8000/health` returns HTTP 200
- [ ] `openm validate` passes all checks
- [ ] `uv run inference.py` completes without exceptions and produces `[START]/[STEP]/[END]` logs
- [ ] All 3 tasks appear in `[SUMMARY]` output
- [ ] At least one task has `final_reward > 0.0` (rewards are not all zero)
- [ ] Rewards vary across tasks (not all the same value)
- [ ] HuggingFace Space shows "Running" status
- [ ] `https://<username>-recruitment-screening-env.hf.space/health` returns 200
- [ ] Submission URL is entered on the dashboard

---

## Troubleshooting Quick Reference

| Problem | Fix |
|---------|-----|
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `openm: command not found` | `pip install "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3"` |
| Docker build fails on `git` step | Network issue — retry or check proxy |
| `port 8000 already in use` | `docker ps` → `docker stop <id>` |
| Space build fails on HuggingFace | Check Space logs: `huggingface.co/spaces/<user>/recruitment-screening-env` → Logs tab |
| All rewards = 0.0 in inference | LLM not formatting decision correctly — see CLAUDE.md §11 |
| `HF_TOKEN` error in inference | `export HF_TOKEN=hf_<your_token>` |
| `No task loaded` error | `reset()` not called before `get_task()` — check inference.py flow |
| Space URL 404 | Wait 5–10 min for Space to build; check Logs tab on HuggingFace |
| `openm validate` fails on yaml | Check `openenv.yaml` — must have `spec_version: 1` |

---

## Useful Commands Reference

```bash
# Check all files are present
ls -la /Users/gopalsaraf/Projects/recruitment-screening-env/
ls -la /Users/gopalsaraf/Projects/recruitment-screening-env/server/

# Check Docker image exists
docker images | grep recruitment

# Check if container is running
docker ps | grep recruitment

# View container logs
docker logs rse-test

# Stop and remove all containers
docker stop $(docker ps -q) && docker rm $(docker ps -aq)

# Run inference with verbose stderr
uv run inference.py 2>&1 | tee inference_output.log

# Check Space status via HF CLI
huggingface-cli repo info <your-username>/recruitment-screening-env --repo-type space

# View deployed Space logs
# Go to: https://huggingface.co/spaces/<username>/recruitment-screening-env → Logs
```

---

## Key URLs

| Resource | URL |
|----------|-----|
| Hackathon Dashboard | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard |
| HuggingFace Spaces | https://huggingface.co/spaces |
| OpenEnv GitHub | https://github.com/meta-pytorch/OpenEnv |
| OpenEnv Catalog | https://meta-pytorch.org/OpenEnv/environments |
| HF Token Settings | https://huggingface.co/settings/tokens |
| Discord (help) | https://discord.gg/Dedhy5pkWD |
| Help Email | help_openenvhackathon@scaler.com |
