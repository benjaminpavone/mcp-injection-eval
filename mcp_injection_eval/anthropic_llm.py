"""Native Claude backend (Anthropic SDK).

The agent loop in agent.py speaks the OpenAI chat format, shared by every provider so the
benchmark harness is identical across models. This adapter translates at the boundary:
OpenAI-style messages in, Messages API request out, OpenAI-style assistant message back.

Why not Anthropic's OpenAI-compatible endpoint: current Claude models think by default and
their thinking blocks must be passed back unchanged on the next turn; the native API keeps
them (stored here under a private key of the returned message), and it rejects a
non-default temperature, which the OpenAI path sends.
"""

from __future__ import annotations

import json

import anthropic

from .llm import QuotaExhausted

RAW = "_anthropic_content"  # raw content blocks (incl. thinking) replayed verbatim
MAX_TOKENS = 16000


def _tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


def _assistant_content(msg: dict) -> list:
    if RAW in msg:
        return msg[RAW]
    blocks: list[dict] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        args = json.loads(tc["function"].get("arguments") or "{}")
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args})
    return blocks


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI chat messages -> (system, Messages API messages)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            # All results of one assistant turn go back in a single user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": _assistant_content(m)})
        else:
            out.append({"role": "user", "content": m["content"]})
    return system, out


def from_anthropic(response) -> dict:
    """Messages API response -> OpenAI-style assistant message (raw blocks kept for replay)."""
    text = "".join(b.text for b in response.content if b.type == "text")
    tool_calls = [
        {"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
        for b in response.content
        if b.type == "tool_use"
    ]
    if response.stop_reason == "refusal":
        text = text or "[refusal]"
    msg: dict = {"role": "assistant", "content": text or None, RAW: response.content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class AnthropicLLM:
    def __init__(self, name: str, api_key: str, model: str, timeout: float = 120.0):
        self.name = name
        self.model = model
        # The SDK retries 429/5xx/connection errors itself, honouring retry-after.
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=5, timeout=timeout)

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        system, msgs = to_anthropic(messages)
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=msgs,
                tools=_tools(tools),
            )
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e).lower():  # prepaid credit used up: stop this model
                raise QuotaExhausted(str(e)) from e
            raise
        return from_anthropic(response)
