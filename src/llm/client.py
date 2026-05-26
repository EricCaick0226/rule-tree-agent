from __future__ import annotations

import json
import os
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://api.example.com/v1"
DEFAULT_MODEL = "your-model-name"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class LLMResponse:
    content: str
    model: str
    raw: dict[str, Any]


class OpenAICompatibleLLMClient:
    """Small OpenAI-compatible chat client.

    The agent uses this client only as a document-grounded proposal engine.
    All generated business content still goes through evidence conversion and
    grounding validation before export.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool | None = None,
    ) -> None:
        _load_dotenv_if_available()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("LLM_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set LLM_API_KEY in .env or export LLM_API_KEY before running."
            )
        self.timeout = float(timeout or os.getenv("LLM_TIMEOUT") or 90)
        self.temperature = float(temperature or os.getenv("LLM_TEMPERATURE") or 0.1)
        self.max_tokens = int(max_tokens or os.getenv("LLM_MAX_TOKENS") or 12000)
        self.stream = _env_bool("LLM_STREAM") if stream is None else stream

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool | None = None,
        disable_thinking: bool = False,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        request_messages = messages
        if disable_thinking:
            request_messages = [
                *messages,
                {
                    "role": "user",
                    "content": "本次请求请直接输出最终 JSON，不要输出思考过程。/no_think",
                },
            ]
        use_stream = self.stream if stream is None else stream
        payload = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if use_stream:
            payload["stream"] = True
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if use_stream:
                    return self._read_streaming_response(response)
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        except (http.client.RemoteDisconnected, TimeoutError, OSError) as exc:
            raise RuntimeError(f"LLM connection failed: {exc}") from exc

        raw = json.loads(body)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response does not contain choices.")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            raise RuntimeError("LLM response content is empty.")
        return LLMResponse(content=content, model=raw.get("model", self.model), raw=raw)

    def _read_streaming_response(self, response: Any) -> LLMResponse:
        content_parts: list[str] = []
        raw_events: list[dict[str, Any]] = []
        model = self.model

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            raw_events.append(event)
            model = event.get("model") or model
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                content_parts.append(content)

        content = "".join(content_parts).strip()
        if not content:
            raise RuntimeError("LLM streaming response content is empty.")
        return LLMResponse(content=content, model=model, raw={"stream": raw_events})

    def generate(self, prompt: str) -> str:
        response = self.chat([{"role": "user", "content": prompt}])
        return response.content
