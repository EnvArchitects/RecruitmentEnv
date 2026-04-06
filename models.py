# Copyright (c) Gopal Saraf. All rights reserved.
# BSD-style license.

"""
Data models for the Recruitment Screening Environment.

The agent interacts exclusively through MCP tools, so Action/Observation
are the standard MCP types re-exported here for pip-installable client use.
"""

from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

__all__ = ["CallToolAction", "CallToolObservation"]
