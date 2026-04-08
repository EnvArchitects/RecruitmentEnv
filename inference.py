"""
Recruitment Screening Environment — Baseline Inference Script.

Runs three graded episodes (easy → medium → hard) using an LLM agent via the
Hugging Face Router. Emits structured [START] / [STEP] / [END] logs to stdout
in the exact format required by the hackathon evaluation harness.

Prerequisites:
    1. Build the Docker image:
       docker build -t recruitment-screening-env .

    2. Set your HF token:
       export HF_TOKEN=hf_...

    3. Run:
       uv run inference.py

    Or with overrides:
       DOCKER_IMAGE=recruitment-screening-env \\
       MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \\
       HF_TOKEN=hf_... \\
       uv run inference.py

Environment variables:
    DOCKER_IMAGE   Docker image name          (default: recruitment-screening-env)
    API_BASE_URL   LLM API endpoint           (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier           (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Hugging Face token         (required)
    ENV_BASE_URL   Override env server URL    (default: auto-start via Docker)
    MAX_STEPS      Max tool calls per episode (default: 6)
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3",
#   "openai>=1.0.0",
#   "requests>=2.31.0",
# ]
# ///

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "recruitment-screening-env")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "6"))

# Three tasks — one per difficulty level
TASKS = [
    {"difficulty": "easy",   "seed": 42},
    {"difficulty": "medium", "seed": 17},
    {"difficulty": "hard",   "seed": 99},
]

# ── Structured log helpers ───────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_start(task_name: str, env: str, model: str) -> None:
    print(f"[START] task={task_name} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )



# ── Docker helpers ───────────────────────────────────────────────────────────

def start_container(image: str, port: int = 8000) -> str | None:
    try:
        r = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:8000", image],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            # print(f"[WARN] docker run failed: {r.stderr.strip()}", file=sys.stderr)
            return None
        cid = r.stdout.strip()
        # print(f"[INFO] Container started: {cid[:12]}", file=sys.stderr)
        return cid
    except Exception as exc:
        # print(f"[WARN] Could not start container: {exc}", file=sys.stderr)
        return None


def wait_for_health(base_url: str, timeout: int = 90) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_container(cid: str) -> None:
    for cmd in (["docker", "stop", cid], ["docker", "rm", cid]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception:
            pass


# ── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert recruitment screening agent for a Financial Technology graduate programme.

You have exactly THREE tools — use them in this order every episode:
  1. get_task()                  → read the candidate package
  2. get_evaluation_criteria()   → read the rubric (optional but recommended)
  3. submit_decision(json_str)   → submit your decision (ends the episode)

═══ SCREENING RULES ═══

MUST-HAVE (automatic No Hire if any violated):
  • No visa sponsorship required
  • Degree completed January 2024 – September 2026
  • Normalized GPA ≥ 3.30 / 4.0   [formula: normalized = (raw / max_scale) × 4.0]
  • Cover letter is specific and genuine (no generic buzzwords)

MUST-NOT-HAVE (automatic No Hire):
  • Work experience > 24 months counted AFTER the latest degree start date
    → EXCLUDE any experience that happened BEFORE the latest degree began
    → If dual degree: use the MOST RECENT degree as the reference point
  • Significant conflict between application_data and resume_text (date diff > 3 months)

OVERALL RATING CODES:
  "4" = Strong Hire  (all must-haves + leadership + technical + initiatives ALL rated 4)
  "3" = Hire         (all must-haves met, no disqualifiers)
  "2" = No Hire      (any must-have violated OR disqualifier present)

ACADEMIC CODES (apply normalization FIRST):
  "4" = normalized_gpa ≥ 3.60
  "3" = 3.40 ≤ normalized_gpa < 3.60
  "2" = 3.30 ≤ normalized_gpa < 3.40
  "1" = normalized_gpa < 3.30  → also triggers automatic No Hire on Overall Rating

WORK EXPERIENCE CODES:
  "4" = ≤ 12 months eligible, high-quality firms
  "3" = ≤ 12 months eligible, relevant
  "2" = no post-degree experience
  "1" = > 24 months eligible  → automatic No Hire

COVER LETTER:
  "1" = passes (specific, genuine, references this company/role)
  "0" = fails (buzzwords, generic praise, could apply to any firm)

═══ DECISION JSON FORMAT ═══

{
  "feedback_responses": {
    "Overall Rating":                        "<2|3|4>",
    "Academic Performance":                  "<1|2|3|4>",
    "Work Experience":                       "<1|2|3|4>",
    "Interest in Finance / Technology":      "<1|2|3|4>",
    "CV Quality":                            "<1|2|3|4>",
    "Passes Cover Letter / Why Us Check":    "<0|1>"
  },
  "justifications": {
    "Overall Rating": "<code> - <Label>: <evidence citing specific numbers>",
    "Academic Performance": "<code>: raw=X/Y → normalized=Z/4.0"
  },
  "executive_summary": [
    "Fact 1 about candidate with specific data",
    "Fact 2 about candidate with specific data"
  ]
}

IMPORTANT: Use only numeric codes in feedback_responses — never text labels.\
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_task",
            "description": "Retrieve the current candidate screening task (resume, application, JD, template).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evaluation_criteria",
            "description": "Retrieve the evaluation rubric with GPA normalization, thresholds, and rules.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": "Submit your hiring decision JSON string. Ends the episode and returns reward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision_json": {
                        "type": "string",
                        "description": (
                            "JSON string with feedback_responses, justifications, "
                            "and executive_summary."
                        ),
                    }
                },
                "required": ["decision_json"],
            },
        },
    },
]


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, llm: OpenAI, task_name: str, difficulty: str, seed: int) -> float:
    """Run one full screening episode. Returns final reward (0.0–1.0)."""

    log_start(task_name=task_name, env="recruitment_screening_env", model=MODEL_NAME)

    # Reset environment for this task
    env.reset(difficulty=difficulty, seed=seed)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"New screening task — difficulty: {difficulty}, seed: {seed}. "
                "Start by calling get_task() to read the candidate profile."
            ),
        },
    ]

    final_reward = 0.0
    step = 0
    done = False
    step_rewards: list[float] = []
    last_error: str | None = None

    while step < MAX_STEPS and not done:
        step += 1
        last_error = None

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=2048,
        )

        choice = response.choices[0]
        assistant_msg: dict = {
            "role":    "assistant",
            "content": choice.message.content or "",
        }
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        messages.append(assistant_msg)

        if not choice.message.tool_calls:
            # Agent stopped calling tools — episode ends
            step_rewards.append(0.0)
            log_step(step=step, action="(no tool call)", reward=0.0, done=True, error=None)
            done = True
            break

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            # Build a compact action string for the log
            args_repr = ", ".join(
                f"{k}={repr(v)[:80]}" for k, v in tool_args.items()
            )
            action_str = f"{tool_name}({args_repr})"

            # Execute tool on the environment
            try:
                result_str = env.call_tool(tool_name, **tool_args)
            except Exception as exc:
                last_error = str(exc)
                result_str = json.dumps({"error": last_error})

            # Track reward if this was a decision submission
            step_reward = 0.0
            if tool_name == "submit_decision":
                try:
                    result_data = json.loads(result_str)
                    step_reward  = float(result_data.get("reward", 0.0))
                    final_reward = step_reward
                    done = bool(result_data.get("done", True))
                    if "error" in result_data:
                        last_error = result_data["error"]
                except Exception as exc:
                    last_error = str(exc)
                    done = True

            step_rewards.append(step_reward)
            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=done,
                error=last_error,
            )

            # Feed result back to LLM
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

    log_end(
        success=(final_reward > 0.0),
        steps=step,
        score=final_reward,
        rewards=step_rewards,
    )
    return final_reward


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        # print("[ERROR] HF_TOKEN is required. Set it via: export HF_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)

    # ── Start environment server ─────────────────────────────────────────────
    container_id = None
    base_url = ENV_BASE_URL

    if not base_url:
        container_id = start_container(DOCKER_IMAGE)
        base_url = "http://localhost:8000"
        if not wait_for_health(base_url, timeout=90):
            # print(
            #     f"[ERROR] Server at {base_url} did not become healthy within 90 seconds.",
            #     file=sys.stderr,
            # )
            if container_id:
                stop_container(container_id)
            sys.exit(1)
        # print(f"[INFO] Server healthy at {base_url}", file=sys.stderr)

    # ── Connect LLM client ───────────────────────────────────────────────────
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # ── Import environment client ────────────────────────────────────────────
    # Import here so PYTHONPATH is resolved after Docker is up
    from client import RecruitmentEnv

    # ── Run all episodes ─────────────────────────────────────────────────────
    total_reward = 0.0
    results: list[dict] = []

    with RecruitmentEnv(base_url=base_url).sync() as env:
        for i, cfg in enumerate(TASKS):
            task_id = f"task_{i + 1}_{cfg['difficulty']}"
            try:
                reward = run_episode(
                    env, llm,
                    task_name=task_id,
                    difficulty=cfg["difficulty"],
                    seed=cfg["seed"],
                )
            except Exception as exc:
                log_end(
                    success=False,
                    steps=0,
                    score=0.0,
                    rewards=[],
                )
                reward = 0.0

            total_reward += reward
            results.append({"task_id": task_id, "difficulty": cfg["difficulty"], "reward": reward})

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if container_id:
        stop_container(container_id)
        # print(f"[INFO] Container {container_id[:12]} stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
