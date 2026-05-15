#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_full_en_zh.sh
# 在同一套模型池下：
#   先跑英文(en)全量 scenarios，再跑中文(zh)全量 scenarios
#
# 断点续跑：
#   SKIP_EXISTING=1 时，已存在的 generation/judge 输出会自动跳过
#
# 可选调参：
#   N_REPEATS=3  (judge重复次数，默认3)
#   JUDGE_MAX_TOKENS=1024
#   GEN_*_MODELS=all|none|自定义列表
#   JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3"
# ============================================================

# 你可以在运行时用环境变量覆盖这些默认值
: "${GEN_OPENAI_MODELS:=all}"
: "${GEN_OLLAMA_MODELS:=all}"
: "${GEN_GROQ_MODELS:=all}"
: "${JUDGE_SPECS:=openai:gpt_5_2,ollama:deepseek_v3}"
: "${N_REPEATS:=3}"
: "${JUDGE_MAX_TOKENS:=1024}"
: "${SKIP_EXISTING:=1}"

echo "================ RUN FULL EN+ZH ================"
echo "[INFO] GEN_OPENAI_MODELS=$GEN_OPENAI_MODELS"
echo "[INFO] GEN_OLLAMA_MODELS=$GEN_OLLAMA_MODELS"
echo "[INFO] GEN_GROQ_MODELS=$GEN_GROQ_MODELS"
echo "[INFO] JUDGE_SPECS=$JUDGE_SPECS"
echo "[INFO] N_REPEATS=$N_REPEATS JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS SKIP_EXISTING=$SKIP_EXISTING"
echo "================================================"

echo "-------------------- [EN] ----------------------"
LANGUAGE_MODE=en \
GEN_OPENAI_MODELS="$GEN_OPENAI_MODELS" \
GEN_OLLAMA_MODELS="$GEN_OLLAMA_MODELS" \
GEN_GROQ_MODELS="$GEN_GROQ_MODELS" \
JUDGE_SPECS="$JUDGE_SPECS" \
N_REPEATS="$N_REPEATS" \
JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
SKIP_EXISTING="$SKIP_EXISTING" \
./scripts/run_multi_generation_and_judge.sh

echo "-------------------- [ZH] ----------------------"
LANGUAGE_MODE=zh \
GEN_OPENAI_MODELS="$GEN_OPENAI_MODELS" \
GEN_OLLAMA_MODELS="$GEN_OLLAMA_MODELS" \
GEN_GROQ_MODELS="$GEN_GROQ_MODELS" \
JUDGE_SPECS="$JUDGE_SPECS" \
N_REPEATS="$N_REPEATS" \
JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
SKIP_EXISTING="$SKIP_EXISTING" \
./scripts/run_multi_generation_and_judge.sh

echo "[DONE] EN + ZH finished."
