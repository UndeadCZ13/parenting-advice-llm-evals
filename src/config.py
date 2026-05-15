# src/config.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================
# Paths
# =========================
DATA_DIR = PROJECT_ROOT / "data"
SCENARIO_DIR = DATA_DIR / "scenarios"
MODEL_OUT_DIR = DATA_DIR / "model_outputs"
JUDGE_OUT_DIR = DATA_DIR / "judge_outputs"

RESULTS_DIR = PROJECT_ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"
MERGED_DIR = RESULTS_DIR / "merged"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


# =========================
# Model Pools (FINAL)
# =========================

# OpenAI models (you said you'll use these three)
OPENAI_MODELS: Dict[str, str] = {
    "gpt_5_2": "gpt-5.2",
    "gpt_5_nano": "gpt-5-nano",
    "gpt_4o_mini": "gpt-4o-mini",
}

# Ollama (Cloud / Local)
OLLAMA_MODEL_REGISTRY: Dict[str, str] = {
    "qwen3_8b": "qwen3:8b",
    "ministral3_14b": "ministral-3:14b-cloud",
    "gpt_oss": "gpt-oss:20b-cloud",
    "deepseek_v3": "deepseek-v3.1:671b-cloud",
    "deepseek_r1": "deepseek-r1:latest",
    "kimi_k2": "kimi-k2-thinking:cloud",
    "minimax_m2": "minimax-m2:cloud",
    "gemini_3":"gemini-3-pro-preview",
    "glm4_9b":"glm4:9b",
    "glm46":"glm-4.6:cloud",
}

# Groq
GROQ_MODEL_REGISTRY: Dict[str, str] = {
    "llama31_8b": "llama-3.1-8b-instant",
    "llama33_70b": "llama-3.3-70b-versatile",
    "qwen3_32b": "qwen/qwen3-32b",
}

SUPPORTED_BACKENDS = ["openai", "ollama", "groq", "local"]


# =========================
# Defaults (feel free to tweak)
# =========================

# Default judge settings (safe defaults)
DEFAULT_BACKEND_JUDGE = "openai"
DEFAULT_OPENAI_MODEL_JUDGE = OPENAI_MODELS["gpt_4o_mini"]
DEFAULT_OLLAMA_MODEL_KEY_JUDGE = "deepseek_v3"  # strong general judge
DEFAULT_GROQ_MODEL_KEY_JUDGE = "llama33_70b"    # strong general judge on Groq

# Default generation model (optional—CLI can override)
DEFAULT_BACKEND_GEN = "openai"
DEFAULT_OPENAI_MODEL_GEN = OPENAI_MODELS["gpt_4o_mini"]
DEFAULT_OLLAMA_MODEL_KEY_GEN = "qwen3_8b"
DEFAULT_GROQ_MODEL_KEY_GEN = "llama31_8b"


# =========================
# Answer Generation Policy (Reasoning-safe, FINAL_ONLY default)
# =========================
ANSWER_STYLE_DEFAULT = "final_only"   # final_only | with_reasoning
STRIP_REASONING = True

# Retry once if suspected truncation (generation-level)
ENABLE_TRUNCATION_RETRY = True
MAX_RETRIES_ON_TRUNCATION = 1
RETRY_TOKEN_MULTIPLIER = 2

# Token budgets (generation-level hint)
MAX_TOKENS_DEFAULT = 1024
MAX_TOKENS_REASONING_MODEL = 2048

# Heuristic: model name contains these => reasoning-heavy
REASONING_MODEL_HINTS = ["thinking", "r1", "reason"]


# =========================
# Utils
# =========================
_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    return _TAG_RE.sub("_", s)[:120]
