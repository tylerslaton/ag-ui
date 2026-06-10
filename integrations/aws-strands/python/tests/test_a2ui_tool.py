"""Unit tests for the AWS Strands A2UI recovery port (OSS-162) — Python.

Mirrors the TypeScript suite
(integrations/aws-strands/typescript/src/__tests__/a2ui-tool.test.ts), covering
the adapter's two DevEx tiers, message-shape helpers, error classification, and
the sub-agent streaming translation:

  Tier 2 (explicit): ``get_a2ui_tools(params)`` returns a Strands
  ``PythonAgentTool`` named ``generate_a2ui`` that runs the toolkit recovery
  loop.

  Tier 1 (auto-inject): ``plan_a2ui_injection(...)`` is the pure per-run
  decision — read the runtime ``injectA2UITool`` flag off ``forwarded_props``,
  infer the model from the wrapped agent, resolve the catalog from
  ``input.context``, and decide whether to inject ``generate_a2ui`` (and which
  injected render tool to drop). Returns ``None`` when it must NOT inject.

String literals mirror the shared constants (``GENERATE_A2UI_TOOL_NAME`` from
ag-ui-a2ui-toolkit, ``RENDER_A2UI_TOOL_NAME`` + ``A2UI_SCHEMA_CONTEXT_DESCRIPTION``
from @ag-ui/a2ui-middleware), hardcoded to keep the suite import-light.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from ag_ui.core import Context, EventType, RunAgentInput, Tool, UserMessage
from strands.tools.registry import ToolRegistry

from ag_ui_strands.a2ui_tool import (
    A2UI_STREAM_KEY,
    classify_a2ui_subagent_error,
    get_a2ui_tools,
    is_auto_injected_a2ui_tool,
    plan_a2ui_injection,
    strands_tool_results_to_agui,
    strip_in_flight_tool_call,
)
from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig

GENERATE_A2UI_TOOL_NAME = "generate_a2ui"
RENDER_A2UI_TOOL_NAME = "render_a2ui"
A2UI_SCHEMA_CONTEXT_DESCRIPTION = (
    "A2UI Component Schema — available components for generating UI surfaces. "
    "Use these component names and properties when creating A2UI operations."
)
A2UI_OPS_KEY = "a2ui_operations"

STUB_MODEL = MagicMock(name="stub-model")
CATALOG = {
    "components": {
        "Row": {"required": ["children"]},
        "HotelCard": {"required": ["name", "rating"]},
    }
}


def _input(forwarded_props=None, context=None, tools=None) -> RunAgentInput:
    return RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        messages=[],
        tools=tools or [],
        context=context or [],
        forwarded_props=forwarded_props or {},
    )


# ---------------------------------------------------------------------------
# Tier 2 — explicit factory
# ---------------------------------------------------------------------------


def test_get_a2ui_tools_default_name():
    tool = get_a2ui_tools({"model": STUB_MODEL})
    assert tool.tool_name == GENERATE_A2UI_TOOL_NAME


def test_get_a2ui_tools_custom_name():
    tool = get_a2ui_tools({"model": STUB_MODEL, "tool_name": "make_ui"})
    assert tool.tool_name == "make_ui"


# ---------------------------------------------------------------------------
# Tier 1 — auto-inject decision
# ---------------------------------------------------------------------------


def test_injects_when_flag_true_and_model_present():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(forwarded_props={"injectA2UITool": True}),
        existing_tool_names=[],
    )
    assert plan is not None
    assert plan["tool_name"] == GENERATE_A2UI_TOOL_NAME
    assert RENDER_A2UI_TOOL_NAME in plan["drop_tool_names"]


def test_drops_custom_named_render_tool_when_flag_is_string():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(forwarded_props={"injectA2UITool": "render_ui_custom"}),
        existing_tool_names=[],
    )
    assert plan is not None
    assert plan["tool_name"] == GENERATE_A2UI_TOOL_NAME
    assert "render_ui_custom" in plan["drop_tool_names"]


def test_skips_and_warns_when_no_model_inferable_orchestrator():
    log = MagicMock()
    plan = plan_a2ui_injection(
        model=None,
        input=_input(forwarded_props={"injectA2UITool": True}),
        existing_tool_names=[],
        log=log,
    )
    assert plan is None
    log.warning.assert_called_once()


def test_no_inject_without_flag_or_override():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(),
        existing_tool_names=[],
    )
    assert plan is None


def test_backend_override_injects_without_runtime_flag():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(),
        existing_tool_names=[],
        config={"inject_a2ui_tool": True},
    )
    assert plan is not None
    assert plan["tool_name"] == GENERATE_A2UI_TOOL_NAME


def test_user_prevails_no_double_inject():
    # THE "USER PREVAILS" REQUIREMENT: explicit dev wiring wins.
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(forwarded_props={"injectA2UITool": True}),
        existing_tool_names=[GENERATE_A2UI_TOOL_NAME],
    )
    assert plan is None


def test_resolves_catalog_from_schema_context_entry():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(
            forwarded_props={"injectA2UITool": True},
            context=[
                Context(
                    description=A2UI_SCHEMA_CONTEXT_DESCRIPTION,
                    value=json.dumps(CATALOG),
                )
            ],
        ),
        existing_tool_names=[],
    )
    assert plan is not None
    assert plan["catalog"] == CATALOG


def test_marker_distinguishes_auto_injected_from_dev_wired():
    plan = plan_a2ui_injection(
        model=STUB_MODEL,
        input=_input(forwarded_props={"injectA2UITool": True}),
        existing_tool_names=[],
    )
    assert plan is not None
    assert is_auto_injected_a2ui_tool(plan["tool"]) is True
    # A dev-wired Tier-2 tool carries no marker.
    assert is_auto_injected_a2ui_tool(get_a2ui_tools({"model": STUB_MODEL})) is False


# ---------------------------------------------------------------------------
# Message-shape helpers (Strands python message dicts)
# ---------------------------------------------------------------------------


def test_strip_in_flight_tool_call_drops_trailing_call():
    messages = [
        {"role": "user", "content": [{"text": "compare hotels"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"name": GENERATE_A2UI_TOOL_NAME, "toolUseId": "t1", "input": {}}}
            ],
        },
    ]
    stripped = strip_in_flight_tool_call(messages, GENERATE_A2UI_TOOL_NAME)
    assert len(stripped) == 1
    assert stripped[0]["role"] == "user"


def test_strip_in_flight_tool_call_keeps_trailing_user_turn():
    messages = [{"role": "user", "content": [{"text": "compare hotels"}]}]
    assert len(strip_in_flight_tool_call(messages, GENERATE_A2UI_TOOL_NAME)) == 1


def test_strands_tool_results_to_agui_reconstructs_a2ui_results():
    envelope = json.dumps({A2UI_OPS_KEY: [{"version": "v0.9"}]})
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tc1",
                        "status": "success",
                        "content": [{"text": envelope}],
                    }
                }
            ],
        }
    ]
    agui = strands_tool_results_to_agui(messages)
    assert len(agui) == 1
    assert agui[0]["role"] == "tool"
    assert agui[0]["tool_call_id"] == "tc1"
    assert A2UI_OPS_KEY in agui[0]["content"]


def test_strands_tool_results_to_agui_handles_json_blocks_and_ignores_non_a2ui():
    # {json} content block form.
    from_json = strands_tool_results_to_agui(
        [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tc2",
                            "status": "success",
                            "content": [{"json": {A2UI_OPS_KEY: [{"version": "v0.9"}]}}],
                        }
                    }
                ],
            }
        ]
    )
    assert len(from_json) == 1
    assert A2UI_OPS_KEY in from_json[0]["content"]
    # Non-A2UI tool results are ignored.
    ignored = strands_tool_results_to_agui(
        [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tc3",
                            "status": "success",
                            "content": [{"text": "just a weather result"}],
                        }
                    }
                ],
            }
        ]
    )
    assert ignored == []


# ---------------------------------------------------------------------------
# Sub-agent error classification
# ---------------------------------------------------------------------------


def test_classify_rethrows_cancellation_and_programmer_errors():
    assert classify_a2ui_subagent_error(asyncio.CancelledError(), False) == "rethrow"
    assert classify_a2ui_subagent_error(Exception("x"), True) == "rethrow"
    assert classify_a2ui_subagent_error(TypeError("x"), False) == "rethrow"
    assert classify_a2ui_subagent_error(NameError("x"), False) == "rethrow"


def test_classify_treats_model_errors_as_recoverable():
    assert classify_a2ui_subagent_error(Exception("Bedrock 429"), False) == "recoverable"


# ---------------------------------------------------------------------------
# Adapter integration — scripted runs (conventions from
# tests/test_streaming_predict_state.py)
# ---------------------------------------------------------------------------


def _template_agent() -> MagicMock:
    mock = MagicMock()
    mock.model = MagicMock()
    mock.system_prompt = "You are helpful"
    mock.tool_registry.registry = {}
    mock.record_direct_tool_call = True
    return mock


def _build_agent(thread_id: str, stream_events: list, config=None) -> StrandsAgent:
    agent = StrandsAgent(
        _template_agent(), name="test-agent", config=config or StrandsAgentConfig()
    )
    mock_inner = MagicMock()
    mock_inner.model = MagicMock()
    mock_inner.tool_registry = ToolRegistry()
    mock_inner.session_manager = None

    async def _stream(_msg):
        for event in stream_events:
            yield event

    mock_inner.stream_async = _stream
    agent._agents_by_thread[thread_id] = mock_inner
    return agent


async def _collect(agent: StrandsAgent, inp: RunAgentInput) -> list:
    return [e async for e in agent.run(inp)]


RENDER_TOOL_INPUT = Tool(
    name=RENDER_A2UI_TOOL_NAME,
    description="render a2ui",
    parameters={"type": "object", "properties": {}},
)


def _msg_input(**overrides) -> RunAgentInput:
    base = dict(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        messages=[UserMessage(id="u1", role="user", content="hi")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    base.update(overrides)
    return RunAgentInput(**base)


@pytest.mark.asyncio
async def test_auto_inject_registers_generate_and_drops_render_across_turns():
    """F1 regression: turn 2 on a cached thread must re-drop the re-synced
    render_a2ui and keep exactly one generate_a2ui (our own marked tool is
    refreshed, never treated as dev-wired)."""
    agent = _build_agent("thread-1", [])
    registry = agent._agents_by_thread["thread-1"].tool_registry

    inp = _msg_input(
        forwarded_props={"injectA2UITool": True}, tools=[RENDER_TOOL_INPUT]
    )
    await _collect(agent, inp)
    names = set(registry.registry.keys())
    assert GENERATE_A2UI_TOOL_NAME in names
    assert RENDER_A2UI_TOOL_NAME not in names

    # Turn 2: syncProxyTools re-adds render_a2ui from input.tools; the hook
    # must drop it again and refresh (not duplicate) generate_a2ui.
    await _collect(agent, inp)
    names = set(registry.registry.keys())
    assert GENERATE_A2UI_TOOL_NAME in names
    assert RENDER_A2UI_TOOL_NAME not in names


@pytest.mark.asyncio
async def test_tool_stream_a2ui_payloads_become_inner_tool_call_events():
    """The generate_a2ui tool yields A2UI_STREAM_KEY payloads; the adapter must
    re-emit them as synthetic inner TOOL_CALL_START/ARGS/END so the middleware
    can drive the building skeleton + progressive paint."""
    events = [
        {
            "tool_stream_event": {
                "data": {
                    A2UI_STREAM_KEY: {
                        "kind": "start",
                        "tool_call_id": "r1",
                        "tool_call_name": RENDER_A2UI_TOOL_NAME,
                    }
                }
            }
        },
        {
            "tool_stream_event": {
                "data": {A2UI_STREAM_KEY: {"kind": "args", "tool_call_id": "r1", "delta": '{"surfaceId":'}}
            }
        },
        {
            "tool_stream_event": {
                "data": {A2UI_STREAM_KEY: {"kind": "args", "tool_call_id": "r1", "delta": '"s1"}'}}
            }
        },
        {
            "tool_stream_event": {
                "data": {A2UI_STREAM_KEY: {"kind": "end", "tool_call_id": "r1"}}
            }
        },
    ]
    agent = _build_agent("thread-1", events)
    out = await _collect(agent, _msg_input())

    starts = [
        e
        for e in out
        if e.type == EventType.TOOL_CALL_START
        and getattr(e, "tool_call_name", None) == RENDER_A2UI_TOOL_NAME
    ]
    assert len(starts) == 1
    assert starts[0].tool_call_id == "r1"

    deltas = [
        getattr(e, "delta", "")
        for e in out
        if e.type == EventType.TOOL_CALL_ARGS and getattr(e, "tool_call_id", None) == "r1"
    ]
    assert "".join(deltas) == '{"surfaceId":"s1"}'

    assert any(
        e.type == EventType.TOOL_CALL_END and getattr(e, "tool_call_id", None) == "r1"
        for e in out
    )
