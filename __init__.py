# Copyright (c) Gopal Saraf. All rights reserved.
# BSD-style license.

"""
Recruitment Screening Environment — root package.

An RL environment where an AI agent learns to screen job candidates across
three difficulty levels, producing structured hiring decisions with rewards
computed from deterministic business rules (no LLM judge required).

MCP Tools:
  get_task()                → candidate resume, application, job description, template
  submit_decision(json_str) → score decision, receive reward (0.0–1.0)
  get_evaluation_criteria() → rubric with GPA normalization rules, thresholds, etc.

Quick start:
    from recruitment_screening_env import RecruitmentEnv

    with RecruitmentEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(difficulty="easy", seed=42)
        task   = env.call_tool("get_task")
        result = env.call_tool("submit_decision", decision_json=...)
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import RecruitmentEnv

__all__ = ["RecruitmentEnv", "CallToolAction", "ListToolsAction"]
