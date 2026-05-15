# src/list_supported_models.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from config import (  # noqa: E402
    SUPPORTED_BACKENDS,
    OPENAI_MODELS,
    OLLAMA_MODEL_REGISTRY,
    GROQ_MODEL_REGISTRY,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_kv(d: Dict[str, Any], indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}- {k}: {v}")


def check_env() -> None:
    print_section("Environment Variables (.env)")
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GROQ_API_KEY",
        "GROQ_BASE_URL",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
    ]
    for k in keys:
        v = os.getenv(k)
        status = "SET" if v else "NOT SET"
        print(f"  {k:20s}: {status}")


def list_ollama_local_models() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if not base_url.startswith("http://localhost"):
        print("  (Not local Ollama; skipping local model listing)")
        return
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            print("  (Failed to query Ollama /api/tags)")
            return
        data = resp.json()
        models = data.get("models", [])
        if not models:
            print("  (No local Ollama models found)")
            return
        print("  Local Ollama models:")
        for m in models:
            name = m.get("name")
            size = m.get("size", "")
            print(f"    - {name} ({size})")
    except Exception as e:
        print(f"  (Error querying local Ollama: {e})")


def main() -> None:
    print_section("Supported Backends")
    for b in SUPPORTED_BACKENDS:
        print(f"  - {b}")

    print_section("OpenAI Model Pool (keys -> model)")
    print_kv(OPENAI_MODELS)

    print_section("Ollama Model Registry (keys -> model)")
    print_kv(OLLAMA_MODEL_REGISTRY)

    print_section("Groq Model Registry (keys -> model)")
    print_kv(GROQ_MODEL_REGISTRY)

    check_env()

    print_section("Local Ollama Model Check")
    list_ollama_local_models()

    print("\nDone.")


if __name__ == "__main__":
    main()
