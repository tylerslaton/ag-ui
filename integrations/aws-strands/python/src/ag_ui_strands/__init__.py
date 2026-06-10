"""
AWS Strands Integration for AG-UI.

Simple adapter following the Agno pattern.
"""
from .agent import StrandsAgent
from .a2ui_tool import (
    A2UI_STREAM_KEY,
    get_a2ui_tools,
    is_auto_injected_a2ui_tool,
    plan_a2ui_injection,
)
from .client_proxy_tool import create_proxy_tool, sync_proxy_tools
from .utils import create_strands_app
from .endpoint import add_strands_fastapi_endpoint, add_ping
from .config import (
    StrandsAgentConfig,
    ToolBehavior,
    ToolCallContext,
    ToolResultContext,
    PredictStateMapping,
    SessionManagerProvider,
)

__all__ = [
    "StrandsAgent",
    "A2UI_STREAM_KEY",
    "get_a2ui_tools",
    "is_auto_injected_a2ui_tool",
    "plan_a2ui_injection",
    "create_proxy_tool",
    "sync_proxy_tools",
    "create_strands_app",
    "add_strands_fastapi_endpoint",
    "add_ping",
    "StrandsAgentConfig",
    "ToolBehavior",
    "ToolCallContext",
    "ToolResultContext",
    "PredictStateMapping",
    "SessionManagerProvider",
]

