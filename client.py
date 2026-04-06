# Copyright (c) Gopal Saraf. All rights reserved.
# BSD-style license.

"""
Recruitment Screening Environment Client.

Provides RecruitmentEnv, which extends MCPToolClient and exposes:
  list_tools()              → discover available MCP tools
  call_tool(name, **kwargs) → invoke a tool by name
  reset(**kwargs)           → start a new screening episode
  step(action)              → low-level step (advanced use)

Quick start (async):
    import asyncio, json
    from recruitment_screening_env import RecruitmentEnv

    async def main():
        async with RecruitmentEnv(base_url="http://localhost:8000") as env:
            await env.reset(difficulty="easy", seed=42)
            task   = json.loads(await env.call_tool("get_task"))
            result = json.loads(
                await env.call_tool("submit_decision", decision_json=json.dumps({
                    "feedback_responses": {
                        "Overall Rating": "3",
                        "Academic Performance": "3",
                        "Work Experience": "3",
                        "Interest in Finance / Technology": "3",
                        "CV Quality": "3",
                        "Passes Cover Letter / Why Us Check": "1",
                    },
                    "justifications": {
                        "Overall Rating": "3 - Hire: GPA 3.55/4.0, no visa issues",
                    },
                    "executive_summary": [
                        "GPA 3.55/4.0 from target university",
                        "6-month relevant internship post-graduation",
                    ],
                }))
            )
            print(f"Reward: {result['reward']}")

    asyncio.run(main())

Sync:
    from recruitment_screening_env import RecruitmentEnv
    import json

    with RecruitmentEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(difficulty="medium", seed=7)
        task   = json.loads(env.call_tool("get_task"))
        result = json.loads(env.call_tool("submit_decision", decision_json=...))
        print(result["reward"])
"""

from openenv.core.mcp_client import MCPToolClient


class RecruitmentEnv(MCPToolClient):
    """
    Client for the Recruitment Screening Environment.

    All functionality is inherited from MCPToolClient.
    Override reset() kwargs supported:
        difficulty: "easy" | "medium" | "hard"
        seed:       int (for reproducibility)
    """
    pass  # MCPToolClient provides all needed functionality
