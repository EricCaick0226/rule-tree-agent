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

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
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

    def generate(self, prompt: str) -> str:
        response = self.chat([{"role": "user", "content": prompt}])
        return response.content
