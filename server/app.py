"""
FastAPI application for the Recruitment Screening Environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Via uv:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .recruitment_environment import RecruitmentEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.recruitment_environment import RecruitmentEnvironment

# Pass the CLASS (not an instance) so each WebSocket session gets its own
# isolated environment instance — required for concurrent training.
app = create_app(
    RecruitmentEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="recruitment_screening_env",
)


def main():
    """Entry point for: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
