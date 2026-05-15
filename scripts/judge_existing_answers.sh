#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# judge_existing_answers.sh
#
# 功能：
#   对一个或多个已生成的 answers(jsonl) 文件，按任意数量 judge 进行评分输出。
#   judge 可来自 openai / ollama / groq，并支持简写 key。
#
# 用法 1（命令行传入多个 answers 文件）：
#   ./scripts/judge_existing_answers.sh "data/model_outputs/a.jsonl" "data/model_outputs/b.jsonl"
#
# 用法 2（不传参数，使用脚本内 ANSWERS_FILES 列表）：
#   ./scripts/judge_existing_answers.sh
#
# 用法 3（只跑一个 judge）：
#   JUDGE_SPECS="openai:gpt_5_2" ./scripts/judge_existing_answers.sh "data/model_outputs/a.jsonl"
#
# 用法 4（增加 Groq judge）：
#   JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3,groq:llama33_70b" \
#     ./scripts/judge_existing_answers.sh "data/model_outputs/a.jsonl"
#
# 可选环境变量：
#   JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3"  (默认：OpenAI gpt-5.2 + Ollama deepseek_v3)
#   N_REPEATS=1                                       (默认：1)
#   MAX_ITEMS=5                                       (默认：不设 -> 全部)
#   JUDGE_MAX_TOKENS=1024                             (默认：1024)
#   OUT_DIR="data/judge_outputs"                      (默认：data/judge_outputs)
#
# Judge spec 格式：
#   backend:model_key_or_name
#     - backend: openai | ollama | groq | local(视为ollama)
#     - model_key_or_name：
#         openai: gpt_5_2 / gpt_5_nano / gpt_4o_mini 或直接写 gpt-5.2
#         ollama: deepseek_v3 / kimi_k2 / ... 或直接写 kimi-k2-thinking:cloud
#         groq: llama31_8b / llama33_70b / qwen3_32b 或直接写 llama-3.3-70b-versatile
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# -----------------------------
# 你也可以在这里列出多个 answers 文件（不传参时会使用它）
# 例：
# ANSWERS_FILES=(
#   "data/model_outputs/zh_openai_gpt-4o-mini.jsonl"
#   "data/model_outputs/zh_ollama_kimi_k2.jsonl"
# )
# -----------------------------
ANSWERS_FILES=(
  # 示例：留空则必须命令行传入
)

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3"
JUDGE_SPECS="${JUDGE_SPECS:-$DEFAULT_JUDGE_SPECS}"

N_REPEATS="${N_REPEATS:-1}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
OUT_DIR="${OUT_DIR:-data/judge_outputs}"

MAX_ITEMS_FLAG=""
if [[ -n "${MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_FLAG="--max-items ${MAX_ITEMS}"
fi


mkdir -p "$OUT_DIR"

# -----------------------------
# Resolvers (shorthand -> normalized)
# -----------------------------
resolve_openai_model() {
  case "$1" in
    gpt_5_2|gpt-5.2) echo "gpt-5.2" ;;
    gpt_5_nano|gpt-5-nano) echo "gpt-5-nano" ;;
    gpt_4o_mini|gpt-4o-mini) echo "gpt-4o-mini" ;;
    *) echo "$1" ;;
  esac
}

normalize_ollama_key_or_name() {
  case "$1" in
    qwen3_8b|qwen3:8b) echo "qwen3_8b" ;;
    ministral3_14b|ministral-3:14b-cloud) echo "ministral3_14b" ;;
    gpt_oss|gpt-oss:20b-cloud) echo "gpt_oss" ;;
    deepseek_v3|deepseek-v3.1:671b-cloud) echo "deepseek_v3" ;;
    deepseek_r1|deepseek-r1:latest) echo "deepseek_r1" ;;
    kimi_k2|kimi-k2-thinking:cloud) echo "kimi_k2" ;;
    minimax_m2|minimax-m2:cloud) echo "minimax_m2" ;;
    *) echo "$1" ;;
  esac
}

normalize_groq_key_or_name() {
  case "$1" in
    llama31_8b|llama-3.1-8b-instant) echo "llama31_8b" ;;
    llama33_70b|llama-3.3-70b-versatile) echo "llama33_70b" ;;
    qwen3_32b|qwen/qwen3-32b) echo "qwen3_32b" ;;
    *) echo "$1" ;;
  esac
}

# -----------------------------
# Judge runner for one answers file
# -----------------------------
run_judges_for_answers() {
  local answers_path="$1"
  local specs="$2"

  IFS=',' read -r -a spec_arr <<< "$specs"
  for spec in "${spec_arr[@]}"; do
    spec="$(echo "$spec" | tr -d '[:space:]')"
    [[ -z "$spec" ]] && continue

    local backend="${spec%%:*}"
    local model_part="${spec#*:}"

    if [[ -z "$backend" || -z "$model_part" || "$backend" == "$spec" ]]; then
      echo "[WARN] Bad judge spec '$spec' (expected backend:model). Skipped."
      continue
    fi

    if [[ "$backend" == "local" ]]; then
      backend="ollama"
    fi

    if [[ "$backend" == "openai" ]]; then
      local openai_model
      openai_model="$(resolve_openai_model "$model_part")"
      local out="$OUT_DIR/$(basename "$answers_path" .jsonl)_judged_openai_${openai_model//\//_}.jsonl"
      echo "[JUDGE] openai:$openai_model | answers=$(basename "$answers_path") -> $(basename "$out")"
      if [[ -n "${MAX_ITEMS:-}" ]]; then
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --openai-model "$openai_model" \
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --max-items "$MAX_ITEMS" \
            --output "$out"
        else
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --openai-model "$openai_model" \
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --output "$out"
        fi


    elif [[ "$backend" == "ollama" ]]; then
      local key
      key="$(normalize_ollama_key_or_name "$model_part")"
      local out="$OUT_DIR/$(basename "$answers_path" .jsonl)_judged_ollama_${key}.jsonl"
      echo "[JUDGE] ollama:$key | answers=$(basename "$answers_path") -> $(basename "$out")"
      if [[ -n "${MAX_ITEMS:-}" ]]; then
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --backend ollama --ollama-model-key "$key"\
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --max-items "$MAX_ITEMS" \
            --output "$out"
        else
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --backend ollama --ollama-model-key "$key" \
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --output "$out"
        fi

    elif [[ "$backend" == "groq" ]]; then
      local key
      key="$(normalize_groq_key_or_name "$model_part")"
      local out="$OUT_DIR/$(basename "$answers_path" .jsonl)_judged_groq_${key}.jsonl"
      echo "[JUDGE] groq:$key | answers=$(basename "$answers_path") -> $(basename "$out")"
      if [[ -n "${MAX_ITEMS:-}" ]]; then
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --backend ollama --ollama-model-key "$key" \
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --max-items "$MAX_ITEMS" \
            --output "$out"
        else
        python -m src.run_judging \
            --answers "$answers_path" \
            --backend openai \
            --backend groq --groq-model-key "$key"\
            --n-repeats "$N_REPEATS" \
            --max-tokens "$JUDGE_MAX_TOKENS" \
            --output "$out"
        fi

    else
      echo "[WARN] Unknown judge backend '$backend' in spec '$spec' (skipped)"
    fi
  done
}

# -----------------------------
# Collect answers files:
#  - If CLI args provided => use them
#  - Else use ANSWERS_FILES in this script
# -----------------------------
FILES_TO_JUDGE=()
if [[ "$#" -gt 0 ]]; then
  for p in "$@"; do
    FILES_TO_JUDGE+=("$p")
  done
else
  FILES_TO_JUDGE=("${ANSWERS_FILES[@]}")
fi

# Filter empty items
FILTERED=()
for p in "${FILES_TO_JUDGE[@]}"; do
  p="$(echo "$p" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$p" ]] && continue
  FILTERED+=("$p")
done
FILES_TO_JUDGE=("${FILTERED[@]}")

if [[ "${#FILES_TO_JUDGE[@]}" -eq 0 ]]; then
  echo "[ERROR] No answers files provided."
  echo "  - Pass file paths as args, OR"
  echo "  - Edit ANSWERS_FILES list inside scripts/judge_existing_answers.sh"
  exit 1
fi

echo "============================================================"
echo "[INFO] Project: $PROJECT_ROOT"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] JUDGE_SPECS=$JUDGE_SPECS"
echo "[INFO] N_REPEATS=$N_REPEATS JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS MAX_ITEMS=${MAX_ITEMS:-ALL}"
echo "[INFO] Answers files (${#FILES_TO_JUDGE[@]}):"
for f in "${FILES_TO_JUDGE[@]}"; do
  echo "  - $f"
done
echo "============================================================"

for answers in "${FILES_TO_JUDGE[@]}"; do
  if [[ ! -f "$answers" ]]; then
    echo "[WARN] File not found (skipped): $answers"
    continue
  fi
  echo "------------------------------------------------------------"
  echo "[INFO] Judging file: $answers"
  run_judges_for_answers "$answers" "$JUDGE_SPECS"
done

echo "============================================================"
echo "[ALL DONE] Judging finished."
echo "Judge outputs: $OUT_DIR"
echo "============================================================"
