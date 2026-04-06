# CLAUDE.md — Recruitment Screening Environment

> **This file is the complete context document for Claude.**
> Load this file at the start of any session to get full project understanding.
> When editing any file in this project, always read this first.

---

## 1. What This Project Is

This is an **OpenEnv-compatible Reinforcement Learning environment** built for the
**Meta PyTorch OpenEnv Hackathon × SST (Scaler School of Technology)**.

**Hackathon deadline: April 8, 2026, 11:59 PM IST.**

The environment teaches an RL agent to screen job candidates for a FinTech graduate
programme. The agent reads a synthetic candidate profile (resume + application form +
job description + feedback template) and produces a structured hiring decision. Rewards
are **fully deterministic** — computed by comparing the agent's decision to ground truth
derived from real production business rules. No LLM judge is used.

### Why this exists / Origin

The environment is directly inspired by `document-screener-ion`, a production AI
recruitment screening system (located at
`/Users/gopalsaraf/Projects/document-screener-ion/`) that integrates with the Lever ATS
and is used by ION Investment Group for graduate programme hiring across USA, UK,
Singapore, and Hong Kong. The business rules, GPA normalization formula, region-specific
criteria, and cover letter quality checks in this environment are all derived from that
production codebase — specifically from:

- `api-server/services/resume_parser/prompts/region_criteria/usa.py`
- `api-server/services/resume_parser/prompts/region_criteria/uk.py`
- `api-server/services/resume_parser/prompts/generate_feedback.py`
- `api-server/services/resume_parser/models.py`
- `api-server/services/resume_parser/llm_feedback_generator.py`

---

## 2. Project Location

```
/Users/gopalsaraf/Projects/recruitment-screening-env/
```

---

## 3. Complete File Structure

```
recruitment-screening-env/
├── CLAUDE.md                            ← YOU ARE HERE — full context document
├── SUBMISSION_STEPS.md                  ← step-by-step hackathon submission guide
├── Dockerfile                           ← multi-stage build (mirrors echo_env exactly)
├── README.md                            ← HuggingFace Space README (with YAML metadata header)
├── __init__.py                          ← root package: exports RecruitmentEnv
├── client.py                            ← RecruitmentEnv(MCPToolClient) — the client class
├── models.py                            ← re-exports CallToolAction, CallToolObservation
├── openenv.yaml                         ← OpenEnv manifest (spec_version, name, port)
├── pyproject.toml                       ← deps + package config
├── inference.py                         ← hackathon inference script (mandatory)
└── server/
    ├── __init__.py                      ← server package init
    ├── app.py                           ← FastAPI entry point via create_app()
    └── recruitment_environment.py       ← ALL core logic (900+ lines)
```

---

## 4. Architecture Deep Dive

### 4.1 How OpenEnv Works (Pattern This Project Uses)

OpenEnv environments follow the **MCPEnvironment** pattern (as of v0.2.3):

```
RecruitmentEnvironment (extends MCPEnvironment)
  └── FastMCP server registered with tools:
        ├── get_task()                → returns candidate JSON
        ├── submit_decision(json_str) → scores decision, returns reward
        └── get_evaluation_criteria() → returns rubric JSON

server/app.py
  └── create_app(RecruitmentEnvironment, CallToolAction, CallToolObservation)
        → FastAPI app with /health, /reset, /step, /state, /ws endpoints

client.py
  └── RecruitmentEnv(MCPToolClient)
        → connects to the server, exposes call_tool(), reset(), list_tools()
```

The agent interacts **exclusively through MCP tools**. It never calls `step()` directly
with custom Action objects — it calls `call_tool("get_task")` etc., which go through
the MCP machinery.

### 4.2 Episode Lifecycle

```
1. env.reset(difficulty="easy"|"medium"|"hard", seed=int)
      → CandidateGenerator.generate(difficulty, seed)
      → stores _current_task, _ground_truth, _metadata
      → returns Observation(done=False, metadata={"status": "ready", ...})

2. env.call_tool("get_task")
      → returns JSON: {resume_text, application_data, job_description, feedback_template, difficulty}

3. env.call_tool("get_evaluation_criteria")   [optional — agent may call this]
      → returns JSON rubric: GPA formula, thresholds, must-haves, etc.

4. env.call_tool("submit_decision", decision_json=<json_str>)
      → compute_reward(decision, ground_truth)
      → returns JSON: {reward, breakdown, done=True}

Episode ends after submit_decision. Call reset() for next episode.
```

### 4.3 The Core Classes (all in server/recruitment_environment.py)

#### `CandidateGenerator`
Generates synthetic candidate profiles with pre-computed ground truth.
- `generate(difficulty, seed)` → `dict` with `candidate_package`, `ground_truth`, `metadata`
- `_easy(rng, seed)` → 3 scenarios: `clear_pass`, `clear_fail_visa`, `clear_fail_gpa`
- `_medium(rng, seed)` → 4 scenarios: `gpa_normalize_pass`, `gpa_normalize_fail`, `dual_degree_exp_excluded`, `exp_near_limit`
- `_hard(rng, seed)` → 4 scenarios: `cover_letter_buzzwords`, `exp_over_disguised`, `strong_hire`, `conflicting_grad_date`
- `_pack(...)` → assembles candidate_package + ground_truth + metadata dict
- `_resume(...)` → builds synthetic resume text string
- `_application(...)` → builds structured application dict

#### `compute_reward(decision, ground_truth) → (float, dict)`
Scores the agent's decision against ground truth. Returns `(reward, breakdown)`.

| Component | Weight | Logic |
|-----------|--------|-------|
| `overall_correct` | 0.30 | `agent["Overall Rating"] == gt["Overall Rating"]` |
| `individual_questions` | 0.30 | 0.06 each × 5 questions (Academic, Work, Interest, CV, CoverLetter) |
| `justification_present` | 0.15 | `any(len(v) > 20 for v in justifications.values())` |
| `violation_handling` | 0.15 | Correctly caught a disqualifier (or no false rejection) |
| `executive_summary` | 0.10 | `len(summary) >= 2` |

All rewards are 0.0–1.0. No partial credit within a question — exact code match required.

#### `RecruitmentEnvironment(MCPEnvironment)`
- `__init__`: creates `FastMCP`, registers 3 tools, calls `super().__init__(mcp)`
- `reset(seed, episode_id, difficulty)`: generates new task, resets state
- `_step_impl`: fallback for non-MCP actions (returns error observation)
- `step` / `step_async`: increments step_count, delegates to super
- `state` property: returns `State(episode_id, step_count)`

---

## 5. Business Rules (The Screening Logic)

These are hard-coded into both `CandidateGenerator` (to set ground truth) and
`get_evaluation_criteria()` (to inform the agent). They come from the production
ION screening system.

### 5.1 MUST-HAVE Requirements (automatic No Hire if violated)

1. **No visa sponsorship required** — `application_data["requires_visa_sponsorship"] == "No"`
2. **Graduation window** — degree completed between January 2024 and September 2026 (inclusive)
3. **Normalized GPA ≥ 3.30/4.0** — formula: `normalized = (raw / max_scale) * 4.0`
4. **Cover letter quality** — must be specific and genuine; generic buzzwords disqualify

### 5.2 MUST-NOT-HAVE Disqualifiers (automatic No Hire)

1. **Work experience > 24 months** — counted **only after the latest degree start date**
   - If dual degree (Bachelor's + Master's): use Master's start date as reference
   - Any experience before the latest degree started is EXCLUDED from the count
2. **Conflicting data** — date difference > 3 months between application and resume,
   or different institution listed in each

### 5.3 GPA Normalization

```
normalized = (raw_gpa / max_scale) * 4.0

Minimum to pass: 3.30 / 4.0

Academic rating codes:
  "4" → normalized >= 3.60  (Outstanding)
  "3" → 3.40 <= normalized < 3.60  (Above Average)
  "2" → 3.30 <= normalized < 3.40  (Average)
  "1" → normalized < 3.30  (Unsatisfactory — also triggers No Hire)
```

### 5.4 Overall Rating Logic

```
"4" = Strong Hire: ALL must-haves met + leadership + technical + initiatives ALL evidenced
                   + every other question rated 4
"3" = Hire:        ALL must-haves met, no disqualifiers
"2" = No Hire:     ANY must-have violated OR any disqualifier present
```

### 5.5 Cover Letter Quality Disqualifiers

Phrases/patterns that trigger a "0" (fails) on the cover letter check:
- "world-class organization", "industry leader"
- Generic motivation: "I am passionate about finance and technology"
- No specific mention of this company's products/research/role
- Brochure/AI-sounding language that could apply to any firm

---

## 6. Difficulty Levels — Scenario Details

### Easy (3 scenarios, selected by `rng.choice`)

| Scenario | What happens | Ground truth |
|----------|-------------|--------------|
| `clear_pass` | GPA 3.50–3.90 / 4.0, no visa, 3–10 months exp, good cover letter | Overall="3" or "4", academic="3"/"4" |
| `clear_fail_visa` | Good GPA but `visa=True` | Overall="2", violated="visa_sponsorship_required" |
| `clear_fail_gpa` | GPA 2.50–3.20 / 4.0 | Overall="2", academic="1", violated="gpa_below_threshold" |

### Medium (4 scenarios)

| Scenario | What happens | Key challenge for agent |
|----------|-------------|------------------------|
| `gpa_normalize_pass` | CGPA 8.3–9.5 / 10.0 | Must normalize: (8.3/10)*4 = 3.32 → passes |
| `gpa_normalize_fail` | CGPA 6.5–8.1 / 10.0 | Must normalize: (6.5/10)*4 = 2.60 → fails |
| `dual_degree_exp_excluded` | Bachelor's 2020 + Master's 2024; 14mo pre-master's work | Must exclude pre-master's exp; only 8mo eligible |
| `exp_near_limit` | 23 months eligible (under 24) | Must not over-trigger No Hire; 23 < 24 = passes |

### Hard (4 scenarios)

| Scenario | What happens | Key challenge for agent |
|----------|-------------|------------------------|
| `cover_letter_buzzwords` | Good GPA/visa but BAD cover letter text | Must catch buzzwords → passes_cl="0", overall="2" |
| `exp_over_disguised` | Resume shows 10mo pre-degree + 16mo post-degree = 26 total | Must exclude pre-degree → only 16mo eligible → passes |
| `strong_hire` | All good-to-haves present, GPA 3.75–3.95 | Must correctly award all 4s → overall="4" |
| `conflicting_grad_date` | App says Jun 2024; resume says May 2025 | Must detect conflict (>3 months diff) → overall="2" |

---

## 7. Feedback Template Structure

The template is the same across all difficulties. It has 7 questions:

```python
FEEDBACK_TEMPLATE = {
    "questions": [
        {"question": "Overall Rating",                     "type": "multiple-choice", "options": ["4 - Strong Hire", "3 - Hire", "2 - No Hire"]},
        {"question": "Academic Performance",               "type": "multiple-choice", "options": ["4 - Outstanding", "3 - Above Average", "2 - Average", "1 - Unsatisfactory"]},
        {"question": "Work Experience",                    "type": "multiple-choice", "options": ["4 - Excellent", "3 - Good", "2 - Limited", "1 - Disqualifying"]},
        {"question": "Interest in Finance / Technology",   "type": "multiple-choice", "options": ["4 - Strong", "3 - Good", "2 - Limited", "1 - None"]},
        {"question": "CV Quality",                         "type": "multiple-choice", "options": ["4 - Outstanding", "3 - Good", "2 - Adequate", "1 - Poor"]},
        {"question": "Passes Cover Letter / Why Us Check", "type": "yes-no"},          # 1=Yes, 0=No
        {"question": "Justification",                      "type": "textarea"},        # free text
    ]
}
```

The agent must return **numeric codes only** in `feedback_responses` (e.g., `"3"`, not
`"3 - Hire"`). The `compute_reward` function does exact string comparison of codes.

---

## 8. Inference Script Details (inference.py)

### What it does
1. Optionally starts a Docker container (`DOCKER_IMAGE`)
2. Waits for `/health` to return 200
3. Connects an `OpenAI` client pointing to the HF Router
4. Runs 3 episodes: easy (seed=42), medium (seed=17), hard (seed=99)
5. Each episode: LLM calls tools in a loop, logs each step, ends when `submit_decision` returns `done=True`
6. Emits `[SUMMARY]` at end

### Log format (mandatory — hackathon evaluator parses this)

```json
{"event": "[START]", "task_id": "task_1_easy", "difficulty": "easy", "seed": 42, "timestamp": "..."}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 1, "tool_name": "get_task", "tool_input": {}, "tool_output_preview": "...", "reward": 0.0, "timestamp": "..."}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 3, "tool_name": "submit_decision", "reward": 0.85, "timestamp": "..."}
{"event": "[END]",   "task_id": "task_1_easy", "difficulty": "easy", "seed": 42, "final_reward": 0.85, "total_steps": 3, "success": true, "timestamp": "..."}
{"event": "[SUMMARY]", "tasks_run": 3, "average_reward": 0.73, "results": [...], "timestamp": "..."}
```

### Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_TOKEN` | (required) | Your HuggingFace token — used as API key for HF Router |
| `DOCKER_IMAGE` | `recruitment-screening-env` | Docker image name |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Any model available on HF Hub |
| `ENV_BASE_URL` | `""` | Override to skip Docker auto-start (e.g., `http://localhost:8000`) |
| `MAX_STEPS` | `6` | Max tool calls per episode before forcing end |

### LLM System Prompt Strategy

The system prompt in `inference.py` (`SYSTEM_PROMPT`) instructs the agent to:
1. Call `get_task()` first
2. Optionally call `get_evaluation_criteria()`
3. Apply GPA normalization, work-exp date math, cover letter check
4. Call `submit_decision(decision_json=<json_str>)` with exact format

The prompt uses `temperature=0.0` for deterministic, reproducible results.

---

## 9. OpenEnv Spec Compliance

This project uses **openenv-core v0.2.3** (pinned in pyproject.toml).

### Required endpoints (auto-provided by `create_app`)
- `GET /health` → 200 OK
- `POST /reset` → calls `env.reset()`
- `POST /step` → calls `env.step()`
- `GET /state` → calls `env.state`
- `GET /web` → web UI (enabled via `ENABLE_WEB_INTERFACE=true` in Dockerfile)
- `WS /ws` → WebSocket for concurrent sessions

### openenv.yaml
```yaml
spec_version: 1
name: recruitment_screening_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### pyproject.toml key points
- Package name: `openenv-recruitment-screening-env`
- Entry point: `server = "recruitment_screening_env.server.app:main"`
- Package directories: `recruitment_screening_env` → `.`, `recruitment_screening_env.server` → `server`

---

## 10. Docker Build

The Dockerfile is a **two-stage build** that mirrors echo_env exactly:

```
Stage 1 (builder):
  FROM ghcr.io/meta-pytorch/openenv-base:latest
  COPY . /app/env
  RUN apt-get install git          # for the openenv-core git dep
  RUN uv sync --no-install-project # cache deps layer
  RUN uv sync                      # install project

Stage 2 (runtime):
  FROM ghcr.io/meta-pytorch/openenv-base:latest
  COPY --from=builder /app/env/.venv /app/.venv
  COPY --from=builder /app/env /app/env
  ENV ENABLE_WEB_INTERFACE=true
  CMD uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Important**: The `uv.lock` file must exist before building. Generate it with `uv lock`.

---

## 11. Common Issues & Fixes

### "Module not found: openenv" during build
→ The `openenv-core` dep uses a git URL. Make sure `git` is installed in the builder stage
(it is, in the Dockerfile). If building fails, check your internet connection during build.

### "No task loaded. Call reset() first."
→ The agent called `get_task()` before `reset()`. In `inference.py`, `env.reset()` is
called before starting the LLM loop. If you're testing manually, always call `reset()` first.

### "Already submitted this episode."
→ `submit_decision` sets `self._submitted = True`. Reset the environment to start fresh.
The LLM agent should not call `submit_decision` twice in one episode.

### Docker container doesn't start
→ Check `docker ps` for existing containers on port 8000. Kill with `docker stop <id>`.
Also ensure the image is built: `docker images | grep recruitment`.

### Inference script runs but all rewards = 0.0
→ The LLM is not submitting in the exact format. Check that:
- `feedback_responses` keys match exactly (case-sensitive, spaces included)
- Values are strings of numeric codes (`"3"` not `3` or `"3 - Hire"`)
- `decision_json` is a valid JSON string (double-encoded as a string argument to the tool)

### HF Push fails
→ Make sure you're logged in: `huggingface-cli login`. Then `openm push <username>/recruitment-screening-env`.

---

## 12. How to Extend This Environment

### Adding a new scenario to a difficulty level
In `CandidateGenerator._easy()` (or `_medium()` / `_hard()`):
1. Add the scenario name to the `rng.choice([...])` list
2. Add an `elif scenario == "new_scenario":` block setting all rating codes and `violated`
3. Make sure `violated` is set correctly for `compute_reward` to score the violation check

### Adding a new question to the feedback template
1. Add the question dict to `FEEDBACK_TEMPLATE["questions"]` in `recruitment_environment.py`
2. Add the corresponding ground truth key in `_pack()` inside `CandidateGenerator`
3. Add it to `SCORED_QUESTIONS` list in `compute_reward` section
4. Update `REWARD_WEIGHTS` if you want to change the weighting
5. Update the `get_evaluation_criteria()` tool's JSON to mention the new question

### Changing the model in inference.py
Set `MODEL_NAME` env var, or edit `MODEL_NAME` at the top of `inference.py`.
Any model on HuggingFace Hub works — it goes through the HF Router.

### Adding a new difficulty level
1. Add it to `DIFFICULTIES = ("easy", "medium", "hard", "expert")` in `RecruitmentEnvironment`
2. Add a `_expert()` method to `CandidateGenerator`
3. Add the routing in `generate()`: `elif difficulty == "expert": return self._expert(rng, seed)`
4. Add the new task to `TASKS` in `inference.py`

---

## 13. Key Design Decisions

**Why MCPEnvironment instead of plain Environment?**
The latest echo_env (OpenEnv v0.2.3) uses `MCPEnvironment` + `FastMCP` tools instead of
the older `Environment` + typed `Action`/`Observation` classes. This gives:
- Automatic tool discovery via `list_tools()`
- Natural language tool descriptions the LLM can read
- Simpler client code (just `call_tool("name", **kwargs)`)

**Why no LLM judge?**
The reward is purely rule-based. This means:
- Rewards are always diverse (different seeds → different scenarios → different ground truths)
- No external API calls required during evaluation
- Hackathon evaluators can trust the score is objective
- The environment runs entirely within the 2 vCPU / 8 GB constraint

**Why synthetic candidates instead of real resumes?**
- No PII, no copyright issues, no data licensing concerns
- Every scenario is precisely controlled and reproducible by seed
- Ground truth is known at generation time — no labelling needed
- Can generate unlimited training episodes cheaply

**Why CGPA/10 as the medium difficulty?**
This is the most common grading system in India (the hackathon's primary audience).
An agent trained on this environment will generalise to real Indian graduate applicants.

---

## 14. Hackathon Context

| Item | Detail |
|------|--------|
| **Hackathon** | Meta PyTorch OpenEnv Hackathon × SST (Scaler School of Technology) |
| **Round 1 deadline** | April 8, 2026, 11:59 PM IST |
| **Submission** | Push to HuggingFace Spaces; submit the Space URL on the dashboard |
| **Judging criteria** | Utility of idea, quality of grader (0.0–1.0 range, diverse), task design, runtime correctness |
| **Infra limits** | Inference script < 20 min; env runs on 2 vCPU / 8 GB RAM |
| **Pre-submission checks** | HF Space deploys, /health returns 200, Dockerfile builds, inference.py runs without errors, 3+ tasks with valid rewards |
| **Multiple submissions** | Allowed; latest one counts |
| **Team lead** | Only the team lead submits |

### Why this environment stands out vs. competition
- Only recruitment-domain environment on HuggingFace (zero competition in this space)
- Inspired by a real production system — not a toy
- CGPA normalization covers the primary hackathon audience (Indian graduates)
- Deterministic rewards that genuinely vary by episode (required by judges)
- Multi-level difficulty with 11 distinct scenarios
- Clean MCP tool design following the latest OpenEnv patterns exactly

---

## 15. File Purposes at a Glance

| File | Purpose | Edit when... |
|------|---------|-------------|
| `server/recruitment_environment.py` | ALL core logic | Changing scenarios, rules, reward weights, tools |
| `inference.py` | Hackathon evaluation script | Changing model, tasks, log format, prompt |
| `server/app.py` | FastAPI wiring | Never (unless OpenEnv API changes) |
| `pyproject.toml` | Dependencies, package config | Adding new deps, renaming package |
| `Dockerfile` | Container build | Adding system deps, changing build strategy |
| `openenv.yaml` | OpenEnv manifest | Changing name or port |
| `client.py` | Python client class | Adding client-side helpers |
| `__init__.py` | Package exports | Adding new exports |
| `models.py` | Type re-exports | Adding new types |
| `README.md` | HuggingFace Space page | Updating docs, HF metadata |
| `CLAUDE.md` | This file — full context | After any significant change |
| `SUBMISSION_STEPS.md` | Step-by-step submission | Updating submission process |
