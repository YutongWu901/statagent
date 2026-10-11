"""LLM clients.

- :class:`DeepSeekClient`: OpenAI-compatible client (DeepSeek by default)
  with JSON-mode output, function-calling support, retries and cost tracking.
- :class:`MockLLM`: deterministic scripted client for tests and pipeline
  smoke runs (never calls the network, never costs money).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from statagent.config import LLMConfig, Pricing


class LLMError(RuntimeError):
    """Raised for provider/parse failures with a user-actionable message."""


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        self.cost_usd += (prompt_tokens * price_in + completion_tokens * price_out) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM reply, tolerating code fences and prose."""
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Response contains no JSON object: {text[:200]!r}") from None
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse JSON from response: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("Expected a JSON object at the top level.")
    return obj


class DeepSeekClient:
    def __init__(self, cfg: LLMConfig, pricing: Pricing | None = None) -> None:
        self.cfg = cfg
        self.pricing = pricing or Pricing()
        self.usage = UsageTracker()
        self._client = None
        if cfg.is_configured:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout, max_retries=0
            )

    @property
    def is_configured(self) -> bool:
        return self.cfg.is_configured and self._client is not None

    def complete(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        if not self.is_configured:
            raise LLMError("No API key configured. Set STATAGENT_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.")
        from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                usage = resp.usage or None
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.usage.add(
                    prompt_tokens, completion_tokens,
                    self.pricing.input_per_mtok, self.pricing.output_per_mtok,
                )
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    }
                    for tc in (msg.tool_calls or [])
                ]
                return LLMResponse(
                    content=msg.content,
                    tool_calls=tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                )
            except RateLimitError as exc:
                raise LLMError(
                    "DeepSeek API rate limit reached. Wait a moment and retry, "
                    "or reduce the request rate."
                ) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
            except APIStatusError as exc:
                if exc.status_code == 401:
                    raise LLMError(
                        "DeepSeek API rejected the key (401). Check "
                        "STATAGENT_DEEPSEEK_API_KEY in your .env file."
                    ) from exc
                raise LLMError(f"DeepSeek API error {exc.status_code}: {exc.message}") from exc
        raise LLMError(f"DeepSeek API unreachable after 3 attempts: {last_exc}")

    def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # DeepSeek requires the word "json" in the prompt for JSON mode.
        if not any("json" in str(m.get("content", "")).lower() for m in messages):
            messages = [dict(messages[0]), *messages[1:]]
            messages[0]["content"] = str(messages[0]["content"]) + "\nRespond with JSON."
        resp = self.complete(messages, json_mode=True)
        if not resp.content:
            raise LLMError("Empty response from the model.")
        return parse_json(resp.content)


class MockLLM:
    """Scripted LLM for tests: no network, deterministic responses, zero cost."""

    def __init__(self) -> None:
        self.scripts: list[
            tuple[Callable[[list[dict[str, Any]]], bool], Callable[[list[dict[str, Any]]], LLMResponse]]
        ] = []
        self.calls: list[dict[str, Any]] = []
        self.usage = UsageTracker()

    def on(
        self,
        predicate: Callable[[list[dict[str, Any]]], bool],
        factory: Callable[[list[dict[str, Any]]], LLMResponse],
    ) -> "MockLLM":
        self.scripts.append((predicate, factory))
        return self

    def complete(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "json_mode": json_mode, "tools": tools})
        for predicate, factory in self.scripts:
            if predicate(messages):
                resp = factory(messages)
                self.usage.add(1, 1, 0.0, 0.0)
                return resp
        return LLMResponse(content="{}")

    def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self.complete(messages, json_mode=True)
        return parse_json(resp.content or "{}")


def create_client(cfg: LLMConfig, pricing: Pricing | None = None) -> DeepSeekClient | None:
    if cfg.is_configured:
        return DeepSeekClient(cfg, pricing)
    return None
