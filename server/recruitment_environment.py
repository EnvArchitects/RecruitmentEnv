"""
Recruitment Screening Environment — core environment logic.

An RL environment where an agent learns to screen job candidates by reading
resumes and application data, then producing structured hiring decisions.

Three difficulty levels:
  easy:   Clear pass/fail, 4.0-scale GPA, no visa complications.
  medium: GPA normalization (CGPA/10), dual-degree, work-exp date math.
  hard:   Conflicting data, cover-letter disqualifiers, Strong-Hire boundary.

All rewards are fully deterministic — no LLM judge required. Ground truth is
computed at reset() time from the same rule-based logic used in production
recruitment systems.

MCP Tools exposed to the agent:
  get_task()                → candidate package + JD + feedback template
  submit_decision(json_str) → scores decision, returns reward + breakdown
  get_evaluation_criteria() → rubric so the agent can self-guide
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

# ---------------------------------------------------------------------------
# Shared feedback template (same structure for all difficulty levels)
# ---------------------------------------------------------------------------

FEEDBACK_TEMPLATE = {
    "title": "Candidate Screening Feedback",
    "instructions": (
        "Evaluate the candidate against the job description and criteria. "
        "For multiple-choice questions use only the numeric prefix. "
        "For yes/no questions return '1' (Yes) or '0' (No). "
        "For text/textarea return a concise evidence-backed response."
    ),
    "questions": [
        {
            "question": "Overall Rating",
            "type": "multiple-choice",
            "description": "Final hire / no-hire recommendation.",
            "options": [
                "4 - Strong Hire: Exceptional — all MUST-HAVEs met + all GOOD-TO-HAVEs demonstrated",
                "3 - Hire: All MUST-HAVEs met, meets standard bar",
                "2 - No Hire: One or more MUST-HAVEs missing or a MUST-NOT-HAVE triggered",
            ],
        },
        {
            "question": "Academic Performance",
            "type": "multiple-choice",
            "description": "Normalized GPA on 4.0 scale. Apply normalization formula first.",
            "options": [
                "4 - Outstanding: normalized GPA >= 3.60",
                "3 - Above Average: normalized GPA 3.40–3.59",
                "2 - Average: normalized GPA 3.30–3.39",
                "1 - Unsatisfactory: normalized GPA < 3.30 (automatic No Hire)",
            ],
        },
        {
            "question": "Work Experience",
            "type": "multiple-choice",
            "description": "Total eligible experience AFTER the latest degree start date.",
            "options": [
                "4 - Excellent: <= 12 months, high-quality internships at recognized firms",
                "3 - Good: <= 12 months, relevant internships",
                "2 - Limited: No post-degree experience",
                "1 - Disqualifying: > 24 months total eligible experience (automatic No Hire)",
            ],
        },
        {
            "question": "Interest in Finance / Technology",
            "type": "multiple-choice",
            "description": "Authentic, evidenced interest in finance or technology.",
            "options": [
                "4 - Strong: Multiple concrete projects or internships showing genuine passion",
                "3 - Good: At least one concrete piece of evidence",
                "2 - Limited: Generic statements, minimal evidence",
                "1 - None: No evidence of interest",
            ],
        },
        {
            "question": "CV Quality",
            "type": "multiple-choice",
            "description": "Structure, clarity, relevance, and up-to-date content.",
            "options": [
                "4 - Outstanding: Exceptionally well-structured, quantified impact, flawless",
                "3 - Good: Clear, structured, relevant",
                "2 - Adequate: Some structure issues but readable",
                "1 - Poor: Unstructured, hard to parse, missing key info",
            ],
        },
        {
            "question": "Passes Cover Letter / Why Us Check",
            "type": "yes-no",
            "description": (
                "Does the cover letter avoid buzzwords and contain specific, genuine "
                "motivation linked to this company/role? Return 1 for Yes (passes), 0 for No."
            ),
        },
        {
            "question": "Justification",
            "type": "textarea",
            "description": (
                "Concise evidence-based summary of your decision. Reference specific "
                "data points (GPA value, graduation date, work experience months, etc.)."
            ),
        },
    ],
}

JOB_DESCRIPTION = """
Position: Graduate Analyst — Financial Technology

We seek exceptional recent graduates for a two-year FinTech graduate programme.
Candidates must:

• Hold a degree completed between January 2024 and September 2026 (inclusive).
• NOT require visa sponsorship.
• Demonstrate authentic interest in finance and/or technology via concrete projects,
  internships, or coursework.
• Have no more than 24 months of professional experience after their latest degree started.
• Maintain a minimum GPA equivalent to 3.30 / 4.00 (all scales normalised using:
  normalized = (raw / max_scale) * 4).

Strong Hire candidates additionally demonstrate leadership, technical depth, and
concrete initiatives with measurable impact.
"""

# ---------------------------------------------------------------------------
# Synthetic candidate generator
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """
    Generates synthetic candidate packages with ground-truth decisions.
    Pure Python — no real resumes or external data needed.
    """

    UNIVERSITIES = [
        ("University of Cambridge", "UK"),
        ("Imperial College London", "UK"),
        ("University of Edinburgh", "UK"),
        ("MIT", "USA"),
        ("Cornell University", "USA"),
        ("NYU Stern", "USA"),
        ("NUS", "Singapore"),
        ("HKUST", "Hong Kong"),
    ]
    COMPANIES = [
        "Goldman Sachs", "Morgan Stanley", "JP Morgan", "Barclays",
        "BlackRock", "Two Sigma", "Jane Street", "Accenture",
    ]
    SKILLS = [
        "Python", "Java", "SQL", "Excel", "VBA", "R", "MATLAB",
        "Tableau", "Bloomberg Terminal", "Git", "Machine Learning",
        "Financial Modelling", "REST APIs", "Data Analysis",
    ]
    MONTH_NAMES = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    MONTH_SHORT = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    GOOD_COVER_LETTERS = [
        (
            "During my internship at {company} I rebuilt the FX settlement "
            "reconciliation pipeline in Python, reducing manual effort by 58 hours "
            "per month. That project made me realise I want to work at the "
            "intersection of markets and software. Your firm's open-source work on "
            "FIX-protocol tooling maps directly to what I want to build, and I have "
            "already contributed a small patch to the order-routing module."
        ),
        (
            "In my second year at {uni} I built a pairs-trading bot in Python that "
            "achieved a Sharpe ratio of 1.4 on backtested EUR/USD data over 2022–2024. "
            "That experience crystallised my interest in quantitative finance. I follow "
            "your research on volatility surface modelling and want to contribute to the "
            "derivatives pricing library your quant team maintains on GitHub."
        ),
    ]
    BAD_COVER_LETTERS = [
        (
            "Your world-class organization is an industry leader in financial services. "
            "I am extremely passionate about finance and technology and would love to "
            "contribute to your dynamic, innovative team. Working at such a prestigious "
            "institution would be a transformative experience for my personal and "
            "professional development."
        ),
        (
            "I am applying because I believe this role will provide me with exposure to "
            "cutting-edge technologies in a collaborative environment. Your firm is "
            "renowned for excellence and I am highly motivated to add significant value "
            "and grow within the organization."
        ),
    ]

    # -----------------------------------------------------------------------

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = random.Random(seed)
        if difficulty == "easy":
            return self._easy(rng, seed)
        elif difficulty == "medium":
            return self._medium(rng, seed)
        else:
            return self._hard(rng, seed)

    # ── Easy ────────────────────────────────────────────────────────────────

    def _easy(self, rng: random.Random, seed: int) -> dict:
        scenario = rng.choice(
            ["clear_pass", "clear_fail_visa", "clear_fail_gpa"])
        uni, country = rng.choice(self.UNIVERSITIES)
        name = self._name(rng)
        grad_year = rng.choice([2024, 2025])
        grad_month = rng.randint(5, 8)
        company = rng.choice(self.COMPANIES)

        if scenario == "clear_pass":
            gpa, scale = round(rng.uniform(3.50, 3.90), 2), 4.0
            visa = False
            exp_months = rng.randint(3, 10)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            overall, academic = "3", ("4" if gpa >= 3.60 else "3")
            experience, interest, cv_q, passes_cl = "3", "3", "3", "1"
            violated = None

        elif scenario == "clear_fail_visa":
            gpa, scale = round(rng.uniform(3.50, 3.85), 2), 4.0
            visa = True
            exp_months = rng.randint(3, 8)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            overall, academic = "2", ("4" if gpa >= 3.60 else "3")
            experience, interest, cv_q, passes_cl = "3", "3", "3", "1"
            violated = "visa_sponsorship_required"

        else:  # clear_fail_gpa
            gpa, scale = round(rng.uniform(2.50, 3.20), 2), 4.0
            visa = False
            exp_months = rng.randint(0, 6)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            overall, academic = "2", "1"
            experience, interest, cv_q, passes_cl = "2", "3", "3", "1"
            violated = "gpa_below_threshold"

        return self._pack(
            rng, name, uni, country, grad_year, grad_month,
            gpa, scale, exp_months, visa, cl,
            overall, academic, experience, interest, cv_q, passes_cl, violated,
            "easy", seed, scenario,
        )

    # ── Medium ──────────────────────────────────────────────────────────────

    def _medium(self, rng: random.Random, seed: int) -> dict:
        scenario = rng.choice([
            "gpa_normalize_pass", "gpa_normalize_fail",
            "dual_degree_exp_excluded", "exp_near_limit",
        ])
        uni, country = rng.choice(self.UNIVERSITIES)
        name = self._name(rng)
        grad_year = rng.choice([2024, 2025, 2026])
        grad_month = rng.randint(4, 9)
        company = rng.choice(self.COMPANIES)

        if scenario == "gpa_normalize_pass":
            # CGPA on 10-point scale — normalizes to just above 3.30
            raw, scale = round(rng.uniform(8.3, 9.5), 1), 10.0
            normalized = round((raw / scale) * 4, 2)
            visa = False
            exp_months = rng.randint(4, 12)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            academic = "4" if normalized >= 3.60 else (
                "3" if normalized >= 3.40 else "2")
            overall, experience, interest, cv_q, passes_cl = "3", "3", "3", "3", "1"
            violated = None

        elif scenario == "gpa_normalize_fail":
            raw, scale = round(rng.uniform(6.5, 8.1), 1), 10.0
            normalized = round((raw / scale) * 4, 2)
            visa = False
            exp_months = rng.randint(0, 8)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            overall, academic = "2", "1"
            experience, interest, cv_q, passes_cl = "2", "3", "3", "1"
            violated = "gpa_below_threshold"

        elif scenario == "dual_degree_exp_excluded":
            # Bachelor's 2020 → 14 months work → Master's 2022 → 8 months internship
            # Agent must exclude the pre-Master's experience → only 8 months eligible
            raw, scale = round(rng.uniform(3.45, 3.85), 2), 4.0
            normalized = raw
            visa = False
            exp_months = 8  # eligible post-latest-degree
            grad_year = rng.choice([2024, 2025])
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            academic = "4" if normalized >= 3.60 else "3"
            overall, experience, interest, cv_q, passes_cl = "3", "3", "3", "3", "1"
            violated = None

        else:  # exp_near_limit — 23 months, just under the 24-month limit
            raw, scale = round(rng.uniform(3.50, 3.80), 2), 4.0
            normalized = raw
            visa = False
            exp_months = 23
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            academic = "4" if normalized >= 3.60 else "3"
            overall, experience, interest, cv_q, passes_cl = "3", "4", "3", "3", "1"
            violated = None

        return self._pack(
            rng, name, uni, country, grad_year, grad_month,
            raw, scale, exp_months, False, cl,
            overall, academic, experience, interest, cv_q, passes_cl, violated,
            "medium", seed, scenario,
            dual_degree=(scenario == "dual_degree_exp_excluded"),
        )

    # ── Hard ────────────────────────────────────────────────────────────────

    def _hard(self, rng: random.Random, seed: int) -> dict:
        scenario = rng.choice([
            "cover_letter_buzzwords",
            "exp_over_disguised",
            "strong_hire",
            "conflicting_grad_date",
        ])
        uni, country = rng.choice(self.UNIVERSITIES)
        name = self._name(rng)
        grad_year = rng.choice([2024, 2025])
        grad_month = rng.randint(4, 8)
        company = rng.choice(self.COMPANIES)

        if scenario == "cover_letter_buzzwords":
            raw, scale = round(rng.uniform(3.55, 3.90), 2), 4.0
            visa = False
            exp_months = rng.randint(4, 12)
            cl = rng.choice(self.BAD_COVER_LETTERS)
            academic = "4" if raw >= 3.60 else "3"
            overall, experience, interest, cv_q, passes_cl = "2", "3", "3", "3", "0"
            violated = "cover_letter_buzzwords"

        elif scenario == "exp_over_disguised":
            # Resume shows 10 months pre-degree + 16 months post-degree = 26 total
            # But only 16 months are eligible → should PASS
            raw, scale = round(rng.uniform(3.40, 3.75), 2), 4.0
            visa = False
            exp_months = 16  # eligible only
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            academic = "4" if raw >= 3.60 else "3"
            overall, experience, interest, cv_q, passes_cl = "3", "4", "3", "3", "1"
            violated = None

        elif scenario == "strong_hire":
            # All good-to-haves present → Strong Hire
            raw, scale = round(rng.uniform(3.75, 3.95), 2), 4.0
            visa = False
            exp_months = rng.randint(6, 14)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            overall, academic, experience, interest, cv_q, passes_cl = "4", "4", "4", "4", "4", "1"
            violated = None

        else:  # conflicting_grad_date
            # App says Jun 2024, resume says May 2025 → significant conflict → No Hire
            raw, scale = round(rng.uniform(3.40, 3.70), 2), 4.0
            visa = False
            exp_months = rng.randint(3, 10)
            cl = rng.choice(self.GOOD_COVER_LETTERS).format(
                uni=uni, company=company)
            grad_year, grad_month = 2024, 6  # app date; resume will show May 2025
            academic = "4" if raw >= 3.60 else "3"
            overall, experience, interest, cv_q, passes_cl = "2", "3", "3", "3", "1"
            violated = "conflicting_graduation_date"

        return self._pack(
            rng, name, uni, country, grad_year, grad_month,
            raw, scale, exp_months, False, cl,
            overall, academic, experience, interest, cv_q, passes_cl, violated,
            "hard", seed, scenario,
            conflict=(scenario == "conflicting_grad_date"),
            strong_hire=(scenario == "strong_hire"),
        )

    # ── Pack result ─────────────────────────────────────────────────────────

    def _pack(
        self, rng, name, uni, country, grad_year, grad_month,
        gpa, scale, exp_months, visa, cover_letter,
        overall, academic, experience, interest, cv_q, passes_cl, violated,
        difficulty, seed, scenario,
        dual_degree=False, conflict=False, strong_hire=False,
    ) -> dict:
        normalized = round((gpa / scale) * 4, 2)
        resume = self._resume(
            rng, name, uni, country, grad_year, grad_month,
            gpa, scale, exp_months, visa,
            dual_degree=dual_degree, conflict=conflict, strong_hire=strong_hire,
        )
        application = self._application(
            name, uni, grad_year, grad_month, gpa, scale, visa, cover_letter, exp_months,
        )
        return {
            "candidate_package": {
                "resume_text": resume,
                "application_data": application,
                "job_description": JOB_DESCRIPTION,
                "feedback_template": FEEDBACK_TEMPLATE,
            },
            "ground_truth": {
                "Overall Rating": overall,
                "Academic Performance": academic,
                "Work Experience": experience,
                "Interest in Finance / Technology": interest,
                "CV Quality": cv_q,
                "Passes Cover Letter / Why Us Check": passes_cl,
                "must_have_violated": violated,
                "normalized_gpa": normalized,
            },
            "metadata": {"difficulty": difficulty, "seed": seed, "scenario": scenario},
        }

    # ── Resume builder ───────────────────────────────────────────────────────

    def _resume(
        self, rng, name, uni, country, grad_year, grad_month,
        gpa, scale, exp_months, visa,
        dual_degree=False, conflict=False, strong_hire=False,
    ) -> str:
        grad_str = f"{self.MONTH_NAMES[grad_month]} {grad_year}"
        # Conflict scenario: resume shows a different date to the application
        resume_grad = f"May {grad_year + 1}" if conflict else grad_str
        gpa_str = self._gpa_str(gpa, scale)
        company = rng.choice(self.COMPANIES)
        skills = rng.sample(self.SKILLS, k=rng.randint(5, 9))
        degree = rng.choice(["MSc Financial Engineering", "BEng Computer Science",
                             "BSc Economics and Finance", "MSc Data Science"])

        edu = (
            f"EDUCATION\n"
            f"{uni} — {country}\n"
            f"{degree}\n"
            f"Graduated: {resume_grad}   |   GPA: {gpa_str}\n"
        )
        if dual_degree:
            old_uni, _ = rng.choice(self.UNIVERSITIES)
            edu += (
                f"\n{old_uni}\n"
                f"BSc Economics\n"
                f"Graduated: May 2020   |   GPA: 3.20 / 4.0\n"
            )

        exp = ""
        if exp_months > 0:
            role = rng.choice([
                "Technology Analyst Intern", "Risk Analyst Intern",
                "Quantitative Research Intern", "Software Engineering Intern",
            ])
            exp = (
                f"\nWORK EXPERIENCE\n"
                f"{company} — {role}\n"
                f"June {grad_year} – {'August' if exp_months <= 3 else 'December'} {grad_year}\n"
                f"  • Built automated data pipelines reducing processing time by 35%\n"
                f"  • Developed Python scripts for trade reconciliation workflows\n"
                f"  • Presented analysis findings to senior management\n"
            )

        strong = ""
        if strong_hire:
            strong = (
                "\nLEADERSHIP & INITIATIVES\n"
                "  • President, FinTech Society (120+ members, organised 8 industry panels)\n"
                "  • Founded algorithmic trading club, managed £10k paper portfolio (Sharpe 1.6)\n"
                "  • 1st place, CFA Institute Research Challenge (national level)\n"
                "\nTECHNICAL PROJECTS\n"
                "  • Pairs Trading Bot: Python, Backtrader — 23% annual return on backtested data\n"
                "  • Credit Risk Model: XGBoost on loan default dataset — AUC 0.91\n"
                "  • Open-source contributor: 3 merged PRs to QuantLib\n"
            )

        visa_line = "Requires Visa Sponsorship" if visa else "Eligible to work without sponsorship"
        return (
            f"{name}\n"
            f"{visa_line}\n"
            f"Email: {name.lower().replace(' ', '.')}@email.com\n\n"
            f"{edu}{exp}\n"
            f"SKILLS\n{', '.join(skills)}\n"
            f"{strong}\n"
            f"ACTIVITIES\n"
            f"  • Member, Investment Banking Society\n"
            f"  • Volunteer financial literacy tutor\n"
        ).strip()

    def _application(
        self, name, uni, grad_year, grad_month, gpa, scale, visa, cover_letter, exp_months,
    ) -> dict:
        return {
            "candidate_name": name,
            "university": uni,
            "degree": "MSc Financial Engineering",
            "graduation_date": f"{self.MONTH_SHORT[grad_month]} {grad_year}",
            "gpa": f"{gpa} / {scale}",
            "requires_visa_sponsorship": "Yes" if visa else "No",
            "months_of_post_degree_work_experience": exp_months,
            "why_our_company": cover_letter,
            "available_to_start": "September 2026",
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _name(self, rng: random.Random) -> str:
        first = rng.choice([
            "Arjun", "Priya", "James", "Emma", "Wei", "Sofia",
            "Mohammed", "Aisha", "Lucas", "Mei", "Oliver", "Sara",
        ])
        last = rng.choice([
            "Sharma", "Chen", "Williams", "Müller", "Nguyen",
            "Patel", "Kim", "Santos", "Okafor", "Ivanova",
        ])
        return f"{first} {last}"

    def _gpa_str(self, gpa: float, scale: float) -> str:
        if scale == 10.0:
            return f"{gpa} / 10.0 (CGPA)"
        return f"{gpa} / {scale}"


# ---------------------------------------------------------------------------
# Reward computation (fully deterministic)
# ---------------------------------------------------------------------------

SCORED_QUESTIONS = [
    "Overall Rating",
    "Academic Performance",
    "Work Experience",
    "Interest in Finance / Technology",
    "CV Quality",
    "Passes Cover Letter / Why Us Check",
]

REWARD_WEIGHTS = {
    "overall_correct":       0.30,
    "individual_questions":  0.30,   # 0.06 each × 5 questions
    "justification_present": 0.15,
    "violation_handling":    0.15,
    "executive_summary":     0.10,
}


def compute_reward(decision: dict, ground_truth: dict) -> tuple[float, dict]:
    """Score agent decision against ground truth. Returns (reward, breakdown)."""
    responses = decision.get("feedback_responses", {})
    reward = 0.0
    breakdown: dict = {}

    # 1. Overall rating (0.30)
    agent_overall = str(responses.get("Overall Rating", "")).strip()
    gt_overall = str(ground_truth.get("Overall Rating", "")).strip()
    overall_ok = agent_overall == gt_overall
    if overall_ok:
        reward += REWARD_WEIGHTS["overall_correct"]
    breakdown["overall_correct"] = {
        "agent": agent_overall, "expected": gt_overall, "ok": overall_ok}

    # 2. Individual questions (0.06 each)
    individual = [q for q in SCORED_QUESTIONS if q != "Overall Rating"]
    per_q = REWARD_WEIGHTS["individual_questions"] / len(individual)
    q_detail: dict = {}
    for q in individual:
        agent_v = str(responses.get(q, "")).strip()
        gt_v = str(ground_truth.get(q, "")).strip()
        ok = agent_v == gt_v
        if ok:
            reward += per_q
        q_detail[q] = {"agent": agent_v, "expected": gt_v, "ok": ok}
    breakdown["individual_questions"] = q_detail

    # 3. Justification present (0.15)
    justifications = decision.get("justifications", {})
    has_justification = (
        isinstance(justifications, dict)
        and any(len(str(v)) > 20 for v in justifications.values())
    )
    if has_justification:
        reward += REWARD_WEIGHTS["justification_present"]
    breakdown["justification_present"] = has_justification

    # 4. Must-have violation handling (0.15)
    violated = ground_truth.get("must_have_violated")
    if violated:
        # Should have said No Hire (2)
        caught = (agent_overall == "2")
        if caught:
            reward += REWARD_WEIGHTS["violation_handling"]
        breakdown["violation_caught"] = {
            "violated": violated, "caught": caught}
    else:
        # No violation — should NOT have said No Hire (no false rejection)
        no_false = (agent_overall != "2")
        if no_false:
            reward += REWARD_WEIGHTS["violation_handling"]
        breakdown["no_false_rejection"] = no_false

    # 5. Executive summary (0.10)
    summary = decision.get("executive_summary", [])
    has_summary = isinstance(summary, list) and len(summary) >= 2
    if has_summary:
        reward += REWARD_WEIGHTS["executive_summary"]
    breakdown["executive_summary_present"] = has_summary

    # Clamp to strictly open interval (0, 1) — platform rejects 0.0 and 1.0 exactly
    _EPS = 0.001
    reward = round(min(max(reward, _EPS), 1.0 - _EPS), 4)
    breakdown["total_reward"] = reward
    return reward, breakdown


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class RecruitmentEnvironment(MCPEnvironment):
    """
    Recruitment Screening Gym.

    MCP tools exposed:
      get_task()                → candidate package
      submit_decision(json_str) → reward + breakdown (ends episode)
      get_evaluation_criteria() → rubric / scoring guide

    Episode flow:
      1. reset(difficulty=..., seed=...)
      2. agent → get_task()
      3. agent → get_evaluation_criteria()    (optional)
      4. agent → submit_decision(json_str)    → reward; done=True
    """

    DIFFICULTIES = ("easy", "medium", "hard")

    def __init__(self):
        self._generator = CandidateGenerator()
        self._current_task: dict = {}
        self._ground_truth: dict = {}
        self._metadata: dict = {}
        self._submitted: bool = False
        self._last_reward: float = 0.001  # Updated from 0.0
        self._episode_count: int = 0

        mcp = FastMCP("recruitment_screening_env")

        # ── Tool: get_task ─────────────────────────────────────────────────
        @mcp.tool
        def get_task() -> str:
            """
            Return the current screening task as JSON.

            The payload contains:
              resume_text        — full candidate resume (text)
              application_data   — structured form answers (GPA, visa, graduation, etc.)
              job_description    — role requirements and hiring criteria
              feedback_template  — questions you must answer with allowed options
              difficulty         — easy | medium | hard
              episode_id         — unique identifier for this episode

            Call this first after each reset().
            """
            if not self._current_task:
                return json.dumps({"error": "No task loaded. Call reset() first."})
            pkg = self._current_task.get("candidate_package", {})
            return json.dumps({
                **pkg,
                "difficulty": self._metadata.get("difficulty", "unknown"),
                "episode_id": self._state.episode_id,
                "screening_hint": (
                    "Normalize GPA: normalized = (raw / max_scale) * 4.0 | "
                    "Count work experience ONLY after the latest degree start date | "
                    "Check cover letter for vague buzzwords (disqualifier) | "
                    "Check for conflicts between application_data and resume_text"
                ),
            })

        # ── Tool: submit_decision ───────────────────────────────────────────
        @mcp.tool
        def submit_decision(decision_json: str) -> str:
            """
            Submit your screening decision. Ends the episode and returns reward.

            Args:
                decision_json: JSON string with structure:
                {
                  "feedback_responses": {
                    "Overall Rating": "3",
                    "Academic Performance": "3",
                    "Work Experience": "3",
                    "Interest in Finance / Technology": "3",
                    "CV Quality": "3",
                    "Passes Cover Letter / Why Us Check": "1"
                  },
                  "justifications": {
                    "Overall Rating": "3 - Hire: GPA 3.52/4.0, no visa issues, 6m internship"
                  },
                  "executive_summary": [
                    "GPA 3.52/4.0 from Cambridge — passes threshold",
                    "Relevant Goldman Sachs internship post-graduation"
                  ]
                }

            Returns:
                JSON with reward (0.0–1.0), per-component breakdown, and done=true.
            """
            if self._submitted:
                return json.dumps({
                    "error": "Already submitted this episode. Call reset() for a new task.",
                    "reward": self._last_reward,
                    "done": True,
                })
            try:
                decision = json.loads(decision_json)
            except json.JSONDecodeError as e:
                # Updated from 0.0 to 0.001
                return json.dumps({"error": f"Invalid JSON: {e}", "reward": 0.001, "done": True})

            reward, breakdown = compute_reward(decision, self._ground_truth)
            self._last_reward = reward
            self._submitted = True

            return json.dumps({
                "reward": reward,
                "breakdown": breakdown,
                "done": True,
                "message": f"Episode complete. Score: {reward:.2f}/1.00. Call reset() for next task.",
            })

        # ── Tool: get_evaluation_criteria ───────────────────────────────────
        @mcp.tool
        def get_evaluation_criteria() -> str:
            """
            Return the evaluation rubric to help you screen candidates correctly.

            Includes: GPA normalization formula, MUST-HAVE requirements,
            MUST-NOT-HAVE disqualifiers, work experience rules, cover letter
            quality check, rating codes, and reward breakdown.
            """
            return json.dumps({
                "gpa_normalization": {
                    "formula": "normalized_gpa = (raw_gpa / max_scale) * 4.0",
                    "minimum_to_pass": 3.30,
                    "examples": [
                        {"raw": 8.5,  "scale": 10.0,
                            "normalized": 3.40, "pass": True},
                        {"raw": 3.55, "scale": 4.0,
                            "normalized": 3.55, "pass": True},
                        {"raw": 6.9,  "scale": 10.0,
                            "normalized": 2.76, "pass": False},
                        {"raw": 75,   "scale": 100,
                            "normalized": 3.00, "pass": False},
                    ],
                },
                "must_have_requirements": [
                    "No visa sponsorship required",
                    "Degree completed between January 2024 and September 2026",
                    "Normalized GPA >= 3.30 / 4.0",
                    "Cover letter passes quality check (specific, not generic buzzwords)",
                ],
                "must_not_have_disqualifiers": [
                    "Work experience > 24 months AFTER latest degree started",
                    "Significant date/institution conflict between application and resume (>3 months diff)",
                ],
                "work_experience_rule": (
                    "Count ONLY experience that started AFTER the latest degree start date. "
                    "Exclude any internships or roles that happened before the latest degree began. "
                    "If candidate has both a Bachelor's and a Master's, use Master's start date."
                ),
                "cover_letter_disqualifiers": [
                    "Buzzwords without evidence: 'world-class', 'industry leader', 'dynamic team'",
                    "Generic motivation that could apply to any firm",
                    "No specific mention of this company's products, research, or role",
                    "AI-sounding or brochure-style promotional language",
                ],
                "overall_rating_logic": {
                    "4 - Strong Hire": (
                        "ALL must-haves met + leadership + technical depth + initiatives "
                        "ALL evidenced + every other question rated 4"
                    ),
                    "3 - Hire": "ALL must-haves met, no disqualifiers",
                    "2 - No Hire": "ANY must-have missing OR any disqualifier present",
                },
                "academic_rating_thresholds": {
                    "4 - Outstanding":    "normalized_gpa >= 3.60",
                    "3 - Above Average":  "3.40 <= normalized_gpa < 3.60",
                    "2 - Average":        "3.30 <= normalized_gpa < 3.40",
                    "1 - Unsatisfactory": "normalized_gpa < 3.30  →  automatic No Hire",
                },
                "reward_weights": REWARD_WEIGHTS,
            }, indent=2)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Generate a new screening task.

        Args:
            seed:       Reproducibility seed. Random if omitted.
            episode_id: Optional episode ID.
            difficulty: "easy" | "medium" | "hard". Cycles if omitted.
        """
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)
        if difficulty not in self.DIFFICULTIES:
            difficulty = self.DIFFICULTIES[self._episode_count % 3]

        self._episode_count += 1
        task = self._generator.generate(difficulty, seed)
        self._current_task = task
        self._ground_truth = task["ground_truth"]
        self._metadata = task["metadata"]
        self._submitted = False
        self._last_reward = 0.001  # Updated from 0.0
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0)

        return Observation(
            done=False,
            reward=0.001,  # Updated from 0.0
            metadata={
                "status": "ready",
                "difficulty": difficulty,
                "seed": seed,
                "episode_id": self._state.episode_id,
                "message": (
                    f"New {difficulty} screening task loaded (seed={seed}). "
                    "Call get_task() to receive the candidate profile."
                ),
            },
        )

    def _step_impl(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False,
            reward=0.001,  # Updated from 0.0
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use MCP tools: get_task(), submit_decision(), get_evaluation_criteria()."
                )
            },
        )

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
