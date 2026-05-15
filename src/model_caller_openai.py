# src/model_caller_openai.py
"""统一的模型调用封装:
- backend="openai": 使用 OpenAI ChatCompletions
- backend="ollama"/"local": 使用本地或 Cloud Ollama (chat 接口)
"""

from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

load_dotenv()


# ========= 公共辅助 =========

def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ========= OpenAI 调用 =========

def call_openai_chat(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 3,
    **_: Any,
) -> Optional[str]:
    """调用 OpenAI / 兼容 OpenAI 的 ChatCompletions 接口。"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY 未设置，无法调用 OpenAI 模型。")
        return None

    base_url = os.getenv("OPENAI_BASE_URL")  # 可选
    client = OpenAI(api_key=api_key, base_url=base_url or None)

    messages = _build_messages(prompt, system_prompt)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] 调用 OpenAI 模型失败，第 {attempt}/{max_retries} 次尝试: {e}")
            if attempt >= max_retries:
                print("[ERROR] OpenAI 多次重试仍失败，本条记录返回 None。")
                return None
            # 简单退避
            sleep_sec = 1.0 * attempt + random.random() * 0.5
            time.sleep(sleep_sec)

    return None


# ========= Ollama 调用 =========

def _ollama_request(
    url: str,
    payload: Dict[str, Any],
    desc: str,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """发送请求; 如果 status!=200, 返回一个带 _error 的 dict, 让上层可以区分 429 等情况。"""
    try:
        resp = requests.post(url, json=payload, timeout=600, headers=headers)
    except Exception as e:
        print(f"[ERROR] 调用 Ollama {desc} 接口失败: {e}")
        return {"_error": True, "status_code": None, "raw_body": str(e)}

    if resp.status_code != 200:
        body = resp.text[:500]
        print(f"[ERROR] Ollama {desc} 接口返回非 200: {resp.status_code}, body={body}")
        return {"_error": True, "status_code": resp.status_code, "raw_body": body}

    try:
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] 解析 Ollama {desc} JSON 失败: {e}, body={resp.text[:500]}")
        return {"_error": True, "status_code": resp.status_code, "raw_body": resp.text[:500]}

    return data


def call_ollama_chat(
    prompt: str,
    model: str = "qwen3:8b",
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 600,          # 目前主要用于文档说明，requests 内部 timeout 固定为 600s
    max_retries: int = 10,        # 🔁 重试次数
    base_delay: float = 1,     # 😴 Cloud 调用基础等待时间（秒）
    **_: Any,
) -> Optional[str]:
    """调用 Ollama /api/chat。
    - 对本地模型和 Cloud 模型都适用
    - 加入：
        - Cloud 调用前的随机 sleep，避免瞬时 QPS 过高导致限流
        - 失败时的重试（最多 max_retries 次），带简单退避
        - 对 Cloud 429 usage limit 的专门友好提示

    使用方式：
    - 本地 Ollama（默认）:
        export OLLAMA_BASE_URL="http://localhost:11434"
    - Ollama Cloud:
        export OLLAMA_BASE_URL="https://ollama.com"
        export OLLAMA_API_KEY="你的 Cloud API Key"
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    api_url = f"{base_url}/api/chat"

    # 构造 Cloud / 本地共用的 messages
    messages = _build_messages(prompt, system_prompt)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    # === 识别是否 Cloud，并准备 Authorization header ===
    headers: Dict[str, str] = {}
    is_cloud = base_url.startswith("https://") and "ollama.com" in base_url

    if is_cloud:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            print("[ERROR] 检测到正在调用 Ollama Cloud (OLLAMA_BASE_URL=https://ollama.com)，"
                  "但未设置环境变量 OLLAMA_API_KEY。Cloud 请求会失败。")
        else:
            # 确保 key 不含奇怪字符，避免 'latin-1' 编码错误
            try:
                api_key.encode("latin-1")
            except UnicodeEncodeError:
                print("[ERROR] OLLAMA_API_KEY 中包含非 latin-1 字符（可能是全角引号、省略号 … 或其它特殊字符）。")
                print("       请在 .env 或环境变量里重新纯文本粘贴 Ollama Cloud 的原始 API key。")
                return None

            headers["Authorization"] = f"Bearer {api_key}"

    last_error_msg: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        # --- Cloud: 每次请求前 sleep 一小会，减少限流 & 抖一下 ---
        if is_cloud:
            # 第一次也稍微等一下，后续重试等得更久一点
            jitter = random.uniform(0.1, 0.3)
            delay = base_delay * (attempt - 1) + jitter
            if delay > 0:
                if attempt == 1:
                    print(f"[INFO] 调用 Ollama Cloud，sleep {delay:.2f}s 以避免瞬时限流。")
                else:
                    print(f"[INFO] Ollama Cloud 重试第 {attempt} 次，sleep {delay:.2f}s 退避。")
                time.sleep(delay)

        # --- 发送请求 ---
        data = _ollama_request(api_url, payload, "/api/chat", headers=headers or None)
        if not data:
            last_error_msg = "[ERROR] _ollama_request 返回空数据。"
        elif data.get("_error"):
            status = data.get("status_code")
            raw_body = data.get("raw_body") or ""
            body_lower = raw_body.lower()

            # 🌟 关键：专门处理 429 usage limit，用友好说明替代空字符串，并不再重试
            if status == 429 and "usage limit" in body_lower:
                msg = (
                    "[Ollama Cloud 429] 已达到当前模型的用量上限，"
                    "请等待额度重置或升级套餐。"
                    "可以改用本地模型（如 qwen3:8b / deepseek-r1:latest）"
                    f" 或减少评测条数。原始返回: {raw_body}"
                )
                print(msg)
                return msg

            # 其他错误，记录后准备重试
            last_error_msg = f"[ERROR] Ollama /api/chat 错误: status={status}, body={raw_body[:200]}"
        else:
            # 正常返回，尝试提取 content
            try:
                message = data.get("message") or {}
                content = message.get("content") or ""
                if content.strip():
                    return content
                else:
                    last_error_msg = "[WARN] Ollama /api/chat 返回空内容。"
            except Exception as e:
                last_error_msg = f"[ERROR] 从 Ollama /api/chat 返回中提取 content 失败: {e}, data={data}"

        # --- 走到这里说明本次调用失败，看看要不要重试 ---
        if attempt < max_retries:
            # 下一轮循环会自动根据 attempt 再 sleep 一次
            print(f"[WARN] Ollama 调用失败，将进行第 {attempt + 1}/{max_retries} 次重试。")
            continue
        else:
            break

    # 多次重试仍失败
    print(f"[ERROR] Ollama 调用在重试 {max_retries} 次后仍失败。最后错误信息：{last_error_msg}")
    return None


# ========= 统一入口 =========

def call_model(
    prompt: str,
    backend: str = "openai",
    model: Optional[str] = None,
    **kwargs: Any,
) -> Optional[str]:
    """统一入口:
    - backend="openai": 走 call_openai_chat
    - backend="ollama"/"local": 走 call_ollama_chat
    """
    backend = backend.lower()

    if model is None:
        if backend == "openai":
            model = "gpt-4o-mini"
        elif backend in {"ollama", "local"}:
            model = "qwen3:8b"
        else:
            raise ValueError(
                f"backend='{backend}' 需要指定 model 参数，或扩展 call_model 中的默认配置。"
            )

    if backend == "openai":
        return call_openai_chat(prompt, model=model, **kwargs)
    elif backend in {"ollama", "local"}:
        return call_ollama_chat(prompt, model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. 支持 'openai', 'ollama', 'local'。")
