# src/model_caller.py
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv

load_dotenv()


class ModelResult(TypedDict, total=False):
    text: str
    finish_reason: Optional[str]
    backend: str
    model: str
    error: Optional[str]


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def call_openai_chat(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 10,
    reasoning_effort: str = "low",
    **_: Any,
) -> ModelResult:
    """
    OpenAI / OpenAI-compatible ChatCompletions.
    Returns dict: {text, finish_reason, backend, model, error}
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"text": "", "finish_reason": None, "backend": "openai", "model": model, "error": "OPENAI_API_KEY not set"}

    base_url = os.getenv("OPENAI_BASE_URL")  # optional gateway
    client = OpenAI(api_key=api_key, base_url=base_url or None)

    model_lc = (model or "").lower()
    is_reasoning = model_lc.startswith("gpt-5") or model_lc.startswith("o3") or model_lc.startswith("o4")

    messages = _build_messages(prompt, system_prompt)

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {"model": model, "messages": messages}

            if is_reasoning:
                kwargs["max_completion_tokens"] = max(2048, int(max_tokens))
                kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = float(temperature)
                kwargs["max_tokens"] = int(max_tokens)

            resp = client.chat.completions.create(**kwargs)

            choice0 = resp.choices[0]
            content = (choice0.message.content or "").strip()
            finish_reason = getattr(choice0, "finish_reason", None)

            return {
                "text": content,
                "finish_reason": finish_reason,
                "backend": "openai",
                "model": model,
                "error": None,
            }

        except Exception as e:
            last_err = str(e)
            print(f"[WARN] OpenAI call failed ({attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return {"text": "", "finish_reason": None, "backend": "openai", "model": model, "error": last_err}
            time.sleep(1.0 * attempt + random.random() * 0.3)

    return {"text": "", "finish_reason": None, "backend": "openai", "model": model, "error": last_err or "unknown"}


def _ollama_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        resp = requests.post(url, json=payload, timeout=600, headers=headers)
    except Exception as e:
        return {"_error": True, "status_code": None, "raw_body": str(e)}

    if resp.status_code != 200:
        return {"_error": True, "status_code": resp.status_code, "raw_body": resp.text[:800]}

    try:
        return resp.json()
    except Exception as e:
        return {"_error": True, "status_code": resp.status_code, "raw_body": f"{e}: {resp.text[:800]}"}


def _extract_ollama_text(data: Dict[str, Any]) -> str:
    """
    Be tolerant to different gateway schemas.
    """
    # native /api/chat
    msg = data.get("message") or {}
    if isinstance(msg, dict):
        c = (msg.get("content") or "").strip()
        if c:
            return c

    # some gateways return 'response'
    c = (data.get("response") or "")
    if isinstance(c, str) and c.strip():
        return c.strip()

    # openai-like
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0] or {}
        m = ch0.get("message") or {}
        if isinstance(m, dict):
            c2 = (m.get("content") or "").strip()
            if c2:
                return c2

    # fallback
    return ""


def _backoff_sleep(attempt: int, base: float = 0.8, cap: float = 12.0) -> None:
    """
    Exponential backoff + jitter.
    attempt starts from 1.
    """
    # attempt=1 -> no sleep; attempt=2 -> base*2^0; attempt=3 -> base*2^1 ...
    if attempt <= 1:
        return
    delay = base * (2 ** (attempt - 2))
    delay = min(cap, delay)
    jitter = random.uniform(0.05, 0.35)
    time.sleep(delay + jitter)


def call_ollama_chat(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 10,
    base_delay: float = 0.8,
    **_: Any,
) -> ModelResult:
    """
    Ollama / Ollama Cloud / Gateway chat.
    Step1: apply exponential backoff on ALL failures (error OR empty content).
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    api_url = f"{base_url}/api/chat"

    headers: Dict[str, str] = {}

    # If using a gateway/cloud that requires key, set it. If not set, let it try (it may still work).
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        # best-effort: only attach if base_url is https or user wants it
        if base_url.startswith("https://") or os.getenv("OLLAMA_ALWAYS_SEND_KEY", "0") == "1":
            headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt),
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
    }

    last_err: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            _backoff_sleep(attempt, base=base_delay, cap=12.0)

        data = _ollama_request(api_url, payload, headers=headers or None)
        if data.get("_error"):
            status = data.get("status_code")
            body = (data.get("raw_body") or "")
            last_err = f"status={status}, body={body[:200]}"
            if attempt < max_retries:
                print(f"[WARN] Ollama call failed, retrying {attempt+1}/{max_retries} ({last_err})")
                continue
            return {"text": "", "finish_reason": None, "backend": "ollama", "model": model, "error": last_err}

        # success response but might be empty content
        content = _extract_ollama_text(data)
        finish_reason = data.get("done_reason")
        if finish_reason is None and data.get("done") is True:
            finish_reason = "stop"

        if content:
            return {"text": content, "finish_reason": finish_reason, "backend": "ollama", "model": model, "error": None}

        last_err = "empty content"
        if attempt < max_retries:
            print(f"[WARN] Ollama call failed, retrying {attempt+1}/{max_retries} ({last_err})")
            continue

    return {"text": "", "finish_reason": None, "backend": "ollama", "model": model, "error": last_err or "unknown"}


def call_model(
    prompt: str,
    backend: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> ModelResult:
    """
    Unified interface. Always returns ModelResult dict.
    """
    b = (backend or "").lower().strip()
    if b == "local":
        b = "ollama"

    if b == "openai":
        return call_openai_chat(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    if b == "ollama":
        return call_ollama_chat(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    return {"text": "", "finish_reason": None, "backend": b, "model": model, "error": f"Unknown backend={backend}. Use openai|ollama|local."}
