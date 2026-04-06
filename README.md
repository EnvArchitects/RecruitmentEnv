---
title: Recruitment Screening Environment
emoji: 🧑‍💼
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv-0.2.3
  - openenv
  - recruitment
  - hiring
  - agent-environment
---

## Hugging Face Space Deployment

This Space is built from OpenEnv environment `recruitment_screening_env`.

**Connecting from code:**
```python
from recruitment_screening_env import RecruitmentEnv
env = RecruitmentEnv(base_url="https://<username>-recruitment-screening-env.hf.space")
```

---

# Recruitment Screening Environment 🧑‍💼

An RL environment where an AI agent learns to screen job candidates — reading
resumes and structured application data, then producing evidence-backed hiring
decisions with rewards computed from **fully deterministic business rules**
(no LLM judge required).

Organizations can configure job descriptions, feedback templates, and hiring
criteria. Agents trained here can autonomously screen thousands of applications
consistently and at scale.

---

## Quick Start

**Async (default):**
```python
import asyncio, json
from recruitment_screening_env import RecruitmentEnv

async def main():
    async with RecruitmentEnv(base_url="http://localhost:8000") as env:
        await env.reset(difficulty="easy", seed=42)

        task = json.loads(await env.call_tool("get_task"))
        print(task["difficulty"])         # "easy"
        print(task["resume_text"][:200])  # candidate resume

        # Optional: read the rubric
        criteria = json.loads(await env.call_tool("get_evaluation_criteria"))

        decision = json.dumps({
            "feedback_responses": {
                "Overall Rating": "3",
                "Academic Performance": "3",
                "Work Experience": "3",
                "Interest in Finance / Technology": "3",
                "CV Quality": "3",
                "Passes Cover Letter / Why Us Check": "1",
            },
            "justifications": {
                "Overall Rating": "3 - Hire: GPA 3.55/4.0 passes threshold, no visa issues",
            },
            "executive_summary": [
                "GPA 3.55/4.0 from target university — above threshold",
                "Relevant 6-month internship at Goldman Sachs post-graduation",
            ],
        })
        result = json.loads(await env.call_tool("submit_decision", decision_json=decision))
        print(f"Reward: {result['reward']}")   # 0.0 – 1.0

asyncio.run(main())
```

**Sync:**
```python
from recruitment_screening_env import RecruitmentEnv
import json

with RecruitmentEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(difficulty="medium", seed=7)
    task   = json.loads(env.call_tool("get_task"))
    result = json.loads(env.call_tool("submit_decision", decision_json=...))
    print(result["reward"])
```

---

## MCP Tools

| Tool | Purpose |
|------|---------|
| `get_task()` | Returns candidate resume, application form, job description, feedback template |
| `get_evaluation_criteria()` | Returns rubric: GPA formula, thresholds, decision rules |
| `submit_decision(decision_json)` | Scores the agent's decision, returns reward 0.0–1.0, ends episode |

### Decision JSON format

```json
{
  "feedback_responses": {
    "Overall Rating":                     "3",
    "Academic Performance":               "3",
    "Work Experience":                    "4",
    "Interest in Finance / Technology":   "3",
    "CV Quality":                         "3",
    "Passes Cover Letter / Why Us Check": "1"
  },
  "justifications": {
    "Overall Rating": "3 - Hire: All MUST-HAVEs met — normalized GPA 3.55, no visa issues",
    "Academic Performance": "3: raw CGPA 8.9/10 → normalized 3.56/4.0 → Above Average"
  },
  "executive_summary": [
    "GPA 3.56/4.0 (normalized) from NUS — passes threshold",
    "Genuine cover letter referencing firm's FIX-protocol tooling — passes quality check"
  ]
}
```

**Rating codes:**

| Question | Codes |
|----------|-------|
| Overall Rating | `4`=Strong Hire · `3`=Hire · `2`=No Hire |
| Academic Performance | `4`≥3.60 · `3` 3.40–3.59 · `2` 3.30–3.39 · `1`<3.30 (auto No Hire) |
| Work Experience | `4`≤12mo high-quality · `3`≤12mo relevant · `2`none · `1`>24mo (auto No Hire) |
| Interest in Finance/Tech | `4`multiple proofs · `3`one proof · `2`generic · `1`none |
| CV Quality | `4`outstanding · `3`good · `2`adequate · `1`poor |
| Passes Cover Letter | `1`=passes · `0`=fails (disqualifying) |

---

## Difficulty Levels

| Level | Scenario | What the agent must handle |
|-------|----------|---------------------------|
| **Easy** | Clear pass or single obvious disqualifier | Spot visa/GPA fail or confirm clean hire on 4.0 scale |
| **Medium** | CGPA/10 normalization, dual-degree, work-exp date math | Apply formula; exclude pre-latest-degree experience |
| **Hard** | Conflicting dates, buzzword cover letters, Strong Hire boundary | Spot contradictions; catch generic language; justify 4/4/4/4 |

---

## Reward Function (Deterministic)

| Component | Weight | Description |
|-----------|--------|-------------|
| Overall rating correct | 0.30 | Hire/No Hire/Strong Hire matches ground truth |
| Individual questions (×5) | 0.30 | 0.06 each — Academic, Work Exp, Interest, CV, Cover Letter |
| Justification present | 0.15 | At least one non-trivial justification (>20 chars) |
| Violation handling | 0.15 | Correctly caught (or didn't false-reject) a disqualifier |
| Executive summary | 0.10 | At least 2 summary bullets present |

**Total: 0.0 – 1.0**

---

## MUST-HAVE Requirements (automatic No Hire if violated)

- No visa sponsorship required
- Degree completed between **January 2024 and September 2026**
- Normalized GPA ≥ **3.30 / 4.0**  (`normalized = (raw / max_scale) × 4`)
- Cover letter passes quality check (specific, genuine, references this role/company)

## MUST-NOT-HAVE Disqualifiers (automatic No Hire)

- Work experience > **24 months** counted **after the latest degree start date**
  (exclude any experience that happened before the latest degree began)
- Significant conflict between application form and resume (date diff > 3 months, different institution)

---

## Building & Running

```bash
# Build Docker image
docker build -t recruitment-screening-env .

# Run locally
docker run -p 8000:8000 recruitment-screening-env

# Verify
curl http://localhost:8000/health
```

## Running the Inference Script

```bash
export HF_TOKEN=hf_...
# Optional:
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

uv run inference.py
```

**Expected stdout (one JSON object per line):**
```
{"event": "[START]", "task_id": "task_1_easy", "difficulty": "easy", "seed": 42, ...}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 1, "tool_name": "get_task", "reward": 0.0, ...}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 2, "tool_name": "get_evaluation_criteria", ...}
{"event": "[STEP]",  "task_id": "task_1_easy", "step": 3, "tool_name": "submit_decision", "reward": 0.85, ...}
{"event": "[END]",   "task_id": "task_1_easy", "difficulty": "easy", "final_reward": 0.85, ...}
{"event": "[START]", "task_id": "task_2_medium", ...}
...
{"event": "[SUMMARY]", "tasks_run": 3, "average_reward": 0.73, ...}
```

---

## Project Structure

```
recruitment-screening-env/
├── __init__.py                          # Package exports (RecruitmentEnv client)
├── client.py                            # RecruitmentEnv client (MCPToolClient subclass)
├── models.py                            # Action / Observation type re-exports
├── openenv.yaml                         # OpenEnv manifest
├── pyproject.toml                       # Dependencies and package metadata
├── Dockerfile                           # Multi-stage container build (mirrors echo_env)
├── inference.py                         # Hackathon inference script ([START]/[STEP]/[END])
├── README.md                            # This file
└── server/
    ├── __init__.py
    ├── app.py                           # FastAPI entry point (create_app)
    └── recruitment_environment.py       # Core: CandidateGenerator, compute_reward, RecruitmentEnvironment
```

---

## Real-World Value

Recruitment screening is a high-volume, high-stakes workflow:

- A trained agent can screen thousands of graduate applications consistently, without fatigue bias
- Configurable hiring criteria (GPA thresholds, visa rules, experience windows, cover letter standards)
- Auditable, evidence-backed decisions with per-question justifications — not black-box scores
- Generalises to new job descriptions and grading scales at inference time

This environment design is directly inspired by a production AI screening system integrated
with the Lever ATS, handling graduate programme applications across USA, UK, Singapore, and
Hong Kong regions with region-specific GPA normalization and visa eligibility rules.
