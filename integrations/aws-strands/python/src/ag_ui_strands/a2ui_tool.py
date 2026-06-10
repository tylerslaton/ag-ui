"""A2UI subagent tool for AWS Strands agents (OSS-162) — Python.

Thin adapter over ``ag-ui-a2ui-toolkit`` — the recovery loop, validation, op
builders, prompt assembly and output envelope all live in the toolkit. This
module owns only the Strands-specific glue (mirrors the TypeScript adapter's
``a2ui-tool.ts``):

  - ``get_a2ui_tools(params, glue=None)`` — Tier 2 (explicit): builds a Strands
    tool the dev adds to their agent's ``tools``. The tool runs the toolkit's
    validate->retry recovery loop, driving a sub-agent that calls
    ``render_a2ui``.
  - ``plan_a2ui_injection(...)`` — Tier 1 (auto-inject): the pure per-run
    decision. Reads the runtime ``injectA2UITool`` flag, infers the model,
    resolves the catalog, threads the run's AG-UI messages + state, and returns
    the tool to register (+ the injected render tool to drop) — or ``None``.

Streaming: the sub-agent's ``render_a2ui`` call must STREAM to the AG-UI wire —
the a2ui middleware's "building" skeleton and progressive paint key off the
inner tool-call's arg deltas, not the final result. The toolkit recovery loop
is synchronous, so it runs in a worker thread; sub-agent stream events are
pushed onto an asyncio queue and re-yielded from the tool's ``stream()`` as
``ToolStreamEvent`` payloads under ``A2UI_STREAM_KEY``, which the adapter
translates into synthetic inner TOOL_CALL_START/ARGS/END events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from strands import Agent
from strands.tools.tools import PythonAgentTool
from strands.types._events import ToolResultEvent, ToolStreamEvent
from strands.types.tools import AgentTool, ToolSpec, ToolUse

from ag_ui.core import RunAgentInput
from ag_ui_a2ui_toolkit import (
    A2UI_OPERATIONS_KEY,
    GENERATE_A2UI_ARG_DESCRIPTIONS,
    GENERATE_A2UI_TOOL_NAME,
    RENDER_A2UI_TOOL_DEF,
    build_a2ui_envelope,
    prepare_a2ui_request,
    resolve_a2ui_tool_params,
    run_a2ui_generation_with_recovery,
    wrap_error_envelope,
)

logger = logging.getLogger("ag_ui_strands")

#: Default name of the render tool the A2UI middleware injects (and we drop).
RENDER_A2UI_TOOL_NAME: str = RENDER_A2UI_TOOL_DEF["function"]["name"]

#: Marker key on ``ToolStreamEvent`` data payloads carrying the sub-agent's
#: render_a2ui streaming progress out of the ``generate_a2ui`` tool. The
#: adapter translates these into synthetic inner TOOL_CALL_START/ARGS/END
#: events on the AG-UI wire. Must match the TypeScript adapter's key.
A2UI_STREAM_KEY = "__a2uiRenderStream"

#: Attribute marking a ``generate_a2ui`` tool this adapter auto-injected
#: (Tier 1), so the per-run hook can tell its OWN prior-turn injection (safe to
#: refresh) apart from a dev-wired Tier-2 tool (USER PREVAILS, never touched).
_A2UI_AUTOINJECT_ATTR = "_a2ui_auto_injected"

#: Context-entry description the ``@ag-ui/a2ui-middleware`` stamps onto the
#: A2UI catalog it injects into ``RunAgentInput.context``. Kept locally so this
#: backend adapter does not depend on the runtime paint-gate package. MUST stay
#: in sync with ``A2UI_SCHEMA_CONTEXT_DESCRIPTION`` in ``@ag-ui/a2ui-middleware``.
A2UI_SCHEMA_CONTEXT_DESCRIPTION = (
    "A2UI Component Schema — available components for generating UI surfaces. "
    "Use these component names and properties when creating A2UI operations."
)


# ---------------------------------------------------------------------------
# Sub-agent error classification
# ---------------------------------------------------------------------------


def classify_a2ui_subagent_error(err: BaseException, aborted: bool) -> str:
    """Classify a sub-agent invoke error. ``"rethrow"`` must unwind the run:

    - cancellation — retrying would defeat the cancel and burn MORE tokens;
    - programmer errors (TypeError/NameError = adapter bugs) — must surface
      loudly, not masquerade as a recoverable "failed attempt".

    ``"recoverable"`` is a genuine model/network error the recovery loop should
    record as a failed attempt (retry or tasteful hard-failure).
    """
    if aborted or isinstance(err, asyncio.CancelledError):
        return "rethrow"
    if isinstance(err, (TypeError, NameError)):
        return "rethrow"
    return "recoverable"


# ---------------------------------------------------------------------------
# Message-shape helpers (Strands python message dicts)
# ---------------------------------------------------------------------------


def _has_tool_use_for(message: dict, tool_name: str) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict):
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict) and tool_use.get("name") == tool_name:
                return True
    return False


def strip_in_flight_tool_call(messages: list, tool_name: str) -> list:
    """Drop the trailing in-flight ``tool_name`` call. When the model invokes
    the generate tool, the assistant turn carrying that toolUse is the last
    message with no matching toolResult yet — passing it to the sub-agent
    (which lacks the tool) is malformed. Only strips when the LAST message is
    that call, so a normal user turn at the tail is preserved."""
    if messages:
        last = messages[-1]
        if (
            isinstance(last, dict)
            and last.get("role") == "assistant"
            and _has_tool_use_for(last, tool_name)
        ):
            return list(messages[:-1])
    return list(messages)


def _tool_result_text(content: Any) -> str:
    """Extract text from a Strands ``toolResult.content`` for A2UI detection.
    Handles raw strings, ``{"text": ...}`` and ``{"json": ...}`` blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif "json" in block:
            parts.append(json.dumps(block["json"]))
    return "".join(parts)


def strands_tool_results_to_agui(messages: list) -> list:
    """Reconstruct the AG-UI ``role:"tool"`` messages the toolkit's
    ``find_prior_surface`` needs (used only for ``intent:"update"``) from
    Strands history. Strands carries tool results as ``toolResult`` blocks
    nested in user turns; emit one AG-UI tool message per result whose content
    contains a prior ``a2ui_operations`` envelope."""
    out: list = []
    fallback_seq = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            result = block.get("toolResult")
            if not isinstance(result, dict):
                continue
            text = _tool_result_text(result.get("content"))
            if not text or A2UI_OPERATIONS_KEY not in text:
                continue
            tool_call_id = result.get("toolUseId")
            if not tool_call_id:
                tool_call_id = f"a2ui-prior-{fallback_seq}"
                fallback_seq += 1
            out.append(
                {
                    "id": tool_call_id,
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": text,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Sub-agent invocation (streaming)
# ---------------------------------------------------------------------------


async def _stream_render_subagent(
    model: Any,
    prompt: str,
    messages: list,
    push: Callable[[dict], None],
) -> Optional[dict]:
    """Run the structured-output sub-agent once: bind a ``render_a2ui`` tool,
    stream the model, push per-event render progress (start / args deltas /
    end) via ``push``, and return the captured ``render_a2ui`` args — or
    ``None`` if the model produced no call."""
    captured: dict | None = None

    def _capture(tool_use: ToolUse, **_kwargs: Any):
        nonlocal captured
        raw = tool_use.get("input")
        captured = raw if isinstance(raw, dict) else {}
        return {
            "toolUseId": tool_use["toolUseId"],
            "status": "success",
            "content": [{"text": "ok"}],
        }

    render_tool = PythonAgentTool(
        tool_name=RENDER_A2UI_TOOL_NAME,
        tool_spec={
            "name": RENDER_A2UI_TOOL_NAME,
            "description": RENDER_A2UI_TOOL_DEF["function"]["description"],
            "inputSchema": {"json": RENDER_A2UI_TOOL_DEF["function"]["parameters"]},
        },
        tool_func=_capture,
    )

    subagent = Agent(
        model=model,
        system_prompt=prompt,
        messages=list(messages),
        tools=[render_tool],
    )

    live_call_id: Optional[str] = None
    emitted_len = 0
    async for event in subagent.stream_async(None):
        if not isinstance(event, dict):
            continue
        current = event.get("current_tool_use")
        if not isinstance(current, dict) or current.get("name") != RENDER_A2UI_TOOL_NAME:
            continue
        call_id = current.get("toolUseId") or "a2ui-render"
        if live_call_id is None:
            live_call_id = call_id
            emitted_len = 0
            push(
                {
                    "kind": "start",
                    "tool_call_id": call_id,
                    "tool_call_name": RENDER_A2UI_TOOL_NAME,
                }
            )
        raw = current.get("input")
        if isinstance(raw, str) and len(raw) > emitted_len:
            push(
                {
                    "kind": "args",
                    "tool_call_id": live_call_id,
                    "delta": raw[emitted_len:],
                }
            )
            emitted_len = len(raw)
    if live_call_id is not None:
        # Some providers deliver the input as a parsed dict (no raw growth); if
        # nothing streamed, emit the captured args as one delta so the
        # middleware still sees the components before the result.
        if emitted_len == 0 and captured:
            push(
                {
                    "kind": "args",
                    "tool_call_id": live_call_id,
                    "delta": json.dumps(captured),
                }
            )
        push({"kind": "end", "tool_call_id": live_call_id})
    return captured


# ---------------------------------------------------------------------------
# The generate_a2ui tool
# ---------------------------------------------------------------------------


class _GenerateA2UITool(AgentTool):
    """Strands tool that delegates A2UI surface generation to a sub-agent
    running the toolkit recovery loop, streaming render progress as it goes."""

    def __init__(self, params: dict, glue: Optional[dict] = None) -> None:
        super().__init__()
        cfg = resolve_a2ui_tool_params(params)
        self._cfg = cfg
        self._glue = glue or {}
        self._spec: ToolSpec = {
            "name": cfg["tool_name"],
            "description": cfg["tool_description"],
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["create", "update"],
                            "description": GENERATE_A2UI_ARG_DESCRIPTIONS["intent"],
                        },
                        "target_surface_id": {
                            "type": "string",
                            "description": GENERATE_A2UI_ARG_DESCRIPTIONS["target_surface_id"],
                        },
                        "changes": {
                            "type": "string",
                            "description": GENERATE_A2UI_ARG_DESCRIPTIONS["changes"],
                        },
                    },
                }
            },
        }

    @property
    def tool_name(self) -> str:
        return self._spec["name"]

    @property
    def tool_spec(self) -> ToolSpec:
        return self._spec

    @property
    def tool_type(self) -> str:
        return "python"

    async def stream(self, tool_use: ToolUse, invocation_state: dict, **kwargs: Any):
        cfg = self._cfg
        glue = self._glue
        raw_input = tool_use.get("input")
        args = raw_input if isinstance(raw_input, dict) else {}
        intent = args.get("intent")
        target_surface_id = args.get("target_surface_id")
        changes = args.get("changes")

        # Strands history for the sub-agent, minus the in-flight generate_a2ui
        # call. Prefer the LIVE calling agent (execution-time history); fall
        # back to the per-thread agent captured at injection time.
        calling_agent = invocation_state.get("agent") or glue.get("strands_agent")
        strands_messages = strip_in_flight_tool_call(
            list(getattr(calling_agent, "messages", None) or []),
            self.tool_name,
        )

        # AG-UI history for the toolkit's find_prior_surface (update intent
        # only): supplied by the adapter on Tier 1, else reconstructed.
        agui_messages = glue.get("agui_messages")
        if agui_messages is None:
            agui_messages = strands_tool_results_to_agui(strands_messages)

        prep = prepare_a2ui_request(
            intent=intent,
            target_surface_id=target_surface_id,
            changes=changes,
            messages=agui_messages,
            state=glue.get("state") or {},
            guidelines=cfg["guidelines"],
        )

        if prep.get("error"):
            envelope = wrap_error_envelope(prep["error"])
        else:
            # The sync recovery loop runs in a worker thread; sub-agent stream
            # progress is pushed onto this queue and re-yielded live.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _push(payload: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            def _invoke_subagent(prompt: str, _attempt: int) -> Optional[dict]:
                # Worker thread: run the async sub-agent on its own loop.
                try:
                    return asyncio.run(
                        _stream_render_subagent(
                            cfg["model"], prompt, strands_messages, _push
                        )
                    )
                except BaseException as err:  # noqa: BLE001 — classified below
                    if classify_a2ui_subagent_error(err, False) == "rethrow":
                        raise
                    logger.warning(
                        "A2UI sub-agent invoke failed; treating as a failed attempt: %s",
                        err,
                    )
                    return None

            def _build_envelope(render_args: dict) -> str:
                return build_a2ui_envelope(
                    args=render_args,
                    is_update=prep["is_update"],
                    target_surface_id=target_surface_id,
                    prior=prep.get("prior"),
                    default_surface_id=cfg["default_surface_id"],
                    default_catalog_id=cfg["default_catalog_id"],
                )

            future = loop.run_in_executor(
                None,
                lambda: run_a2ui_generation_with_recovery(
                    base_prompt=prep["prompt"],
                    catalog=cfg["catalog"],
                    config=cfg["recovery"],
                    on_attempt=cfg["on_a2ui_attempt"],
                    invoke_subagent=_invoke_subagent,
                    build_envelope=_build_envelope,
                ),
            )

            while True:
                while not queue.empty():
                    yield ToolStreamEvent(
                        tool_use, {A2UI_STREAM_KEY: queue.get_nowait()}
                    )
                if future.done():
                    break
                get_task = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {get_task, future}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_task in done:
                    yield ToolStreamEvent(tool_use, {A2UI_STREAM_KEY: get_task.result()})
                else:
                    get_task.cancel()
            envelope = future.result()["envelope"]

        yield ToolResultEvent(
            {
                "toolUseId": tool_use["toolUseId"],
                "status": "success",
                "content": [{"text": envelope}],
            }
        )


def get_a2ui_tools(params: dict, glue: Optional[dict] = None) -> _GenerateA2UITool:
    """Build a Strands tool that delegates A2UI surface generation to a
    sub-agent running the toolkit recovery loop. Add the returned tool to a
    Strands ``Agent``'s ``tools`` list (Tier 2), or let ``plan_a2ui_injection``
    build it (Tier 1)."""
    return _GenerateA2UITool(params, glue)


def is_auto_injected_a2ui_tool(tool: Any) -> bool:
    """True if ``tool`` is a ``generate_a2ui`` this adapter auto-injected."""
    return getattr(tool, _A2UI_AUTOINJECT_ATTR, False) is True


# ---------------------------------------------------------------------------
# Tier 1 — auto-inject decision
# ---------------------------------------------------------------------------


def _resolve_catalog_from_context(input: RunAgentInput) -> Optional[dict]:
    for entry in input.context or []:
        description = getattr(entry, "description", None)
        if description == A2UI_SCHEMA_CONTEXT_DESCRIPTION:
            value = getattr(entry, "value", None)
            if not value:
                return None
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return None
    return None


def plan_a2ui_injection(
    *,
    model: Any,
    input: RunAgentInput,
    existing_tool_names: list,
    config: Optional[dict] = None,
    log: Optional[logging.Logger] = None,
    strands_agent: Any = None,
) -> Optional[dict]:
    """Decide whether to auto-inject ``generate_a2ui`` for this run, mirroring
    the LangGraph contract ("no injectA2UITool, no injection"):

    1. Off unless the runtime forwarded ``injectA2UITool`` (truthy or a custom
       tool-name string) OR a backend ``config["inject_a2ui_tool"]`` override.
    2. USER PREVAILS — a dev-wired ``generate_a2ui`` (Tier 2) is never
       double-injected. (The per-run hook removes our OWN marked tool before
       computing ``existing_tool_names``.)
    3. No inferable model (Graph/Swarm orchestrators) -> warn + skip.
    4. Otherwise build the tool (threading the run's AG-UI messages + state +
       guidelines), resolve the catalog, and drop the injected render tool.

    Returns ``{"tool", "tool_name", "drop_tool_names", "catalog"}`` or ``None``.
    """
    log = log or logger
    config = config or {}

    forwarded = input.forwarded_props or {}
    flag = forwarded.get("injectA2UITool")
    if not flag:
        flag = config.get("inject_a2ui_tool")
    if not flag:
        return None

    tool_name = GENERATE_A2UI_TOOL_NAME
    # USER PREVAILS: explicit dev wiring wins — never double-inject.
    if tool_name in existing_tool_names:
        return None

    if model is None:
        log.warning(
            "A2UI tool injection requested but no model could be inferred from "
            "the agent (multi-agent orchestrators have no model). Skipping "
            "auto-injection — wire get_a2ui_tools() explicitly."
        )
        return None

    render_tool_name = flag if isinstance(flag, str) else RENDER_A2UI_TOOL_NAME
    catalog = config.get("catalog") or _resolve_catalog_from_context(input)

    tool = get_a2ui_tools(
        {
            "model": model,
            "tool_name": tool_name,
            "catalog": catalog,
            "default_catalog_id": config.get("default_catalog_id"),
            "guidelines": config.get("guidelines"),
            "recovery": config.get("recovery"),
        },
        glue={
            "agui_messages": list(input.messages or []),
            "state": input.state,
            "strands_agent": strands_agent,
        },
    )
    setattr(tool, _A2UI_AUTOINJECT_ATTR, True)

    return {
        "tool": tool,
        "tool_name": tool_name,
        "drop_tool_names": [render_tool_name],
        "catalog": catalog,
    }
