#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# run_multi_generation_and_judge.sh
#
# 功能：
#   1) 按配置的 Generation 模型列表（OpenAI/Ollama/Groq）批量生成 answers(jsonl)
#   2) 对每个 answers 文件，按 JUDGE_SPECS（任意数量 judge）批量评分
#
# 关键用法：
#   LANGUAGE_MODE=zh MAX_ITEMS=3 N_REPEATS=1 ./scripts/run_multi_generation_and_judge.sh
#
# 跳过某一类 generation（重要：用 none，而不是空字符串）：
#   GEN_OLLAMA_MODELS=none GEN_GROQ_MODELS=none GEN_OPENAI_MODELS="gpt_4o_mini" ...
#
# 强制用默认全跑：
#   GEN_OLLAMA_MODELS=all GEN_GROQ_MODELS=all GEN_OPENAI_MODELS=all
#
# Judge（任意数量）：
#   JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3,groq:llama33_70b"
#
# 注意：
#   - bash 参数规则里，空字符串会触发默认（:-），所以“跳过”必须显式用 none
# ============================================================

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# -----------------------------
# Language & scenario selection
# -----------------------------
LANGUAGE_MODE="${LANGUAGE_MODE:-en}"

default_scenarios_for_lang() {
  case "$1" in
    zh) echo "data/scenarios/views/zh/parentbench_v0_zh.jsonl" ;;
    en_zh) echo "data/scenarios/views/en_zh/parentbench_v0_en_zh.jsonl" ;;
    en|*) echo "data/scenarios/views/en/parentbench_v0_en.jsonl" ;;
  esac
}

SCENARIOS_FILE="${SCENARIOS_FILE:-$(default_scenarios_for_lang "$LANGUAGE_MODE")}"
if [[ ! -f "$SCENARIOS_FILE" ]]; then
  echo "[ERROR] Scenarios file not found: $SCENARIOS_FILE"
  echo "You can override with: SCENARIOS_FILE=\"...\""
  exit 1
fi

# -----------------------------
# Tuning knobs
# -----------------------------
N_REPEATS="${N_REPEATS:-1}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"   # 1=skip existing outputs (resume), 0=always rerun

MODEL_OUT_DIR="data/model_outputs"
JUDGE_OUT_DIR="data/judge_outputs"
mkdir -p "$MODEL_OUT_DIR" "$JUDGE_OUT_DIR"

# -----------------------------
# Defaults (edit here if you want)
# -----------------------------
DEFAULT_GEN_OPENAI_MODELS="gpt_5_2,gpt_5_nano,gpt_4o_mini"
DEFAULT_GEN_OLLAMA_MODELS="qwen3_8b,ministral3_14b,gpt_oss,deepseek_v3,deepseek_r1,kimi_k2,minimax_m2,glm4_9b,glm46"
DEFAULT_GEN_GROQ_MODELS="llama31_8b,llama33_70b,qwen3_32b"
DEFAULT_JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3"

# -----------------------------
# Read env overrides (support: none | all | custom list)
# -----------------------------
GEN_OPENAI_MODELS_RAW="${GEN_OPENAI_MODELS:-}"
GEN_OLLAMA_MODELS_RAW="${GEN_OLLAMA_MODELS:-}"
GEN_GROQ_MODELS_RAW="${GEN_GROQ_MODELS:-}"

normalize_gen_list() {
  local raw="$1"
  local def="$2"
  local low
  low="$(echo "${raw}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"

  if [[ -z "$raw" ]]; then
    echo "$def"; return
  fi
  if [[ "$low" == "none" || "$low" == "__none__" || "$low" == "skip" || "$low" == "off" ]]; then
    echo "none"; return
  fi
  if [[ "$low" == "all" || "$low" == "__all__" || "$low" == "*" ]]; then
    echo "$def"; return
  fi
  echo "$raw"
}

GEN_OPENAI_MODELS="$(normalize_gen_list "$GEN_OPENAI_MODELS_RAW" "$DEFAULT_GEN_OPENAI_MODELS")"
GEN_OLLAMA_MODELS="$(normalize_gen_list "$GEN_OLLAMA_MODELS_RAW" "$DEFAULT_GEN_OLLAMA_MODELS")"
GEN_GROQ_MODELS="$(normalize_gen_list "$GEN_GROQ_MODELS_RAW" "$DEFAULT_GEN_GROQ_MODELS")"

JUDGE_SPECS="${JUDGE_SPECS:-$DEFAULT_JUDGE_SPECS}"

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
# Progress helpers
# -----------------------------
TOTAL_TASKS=0
DONE_TASKS=0
START_TS="$(date +%s)"

count_list_items() {
  local s="$1"
  local low
  low="$(echo "$s" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  if [[ "$low" == "none" || -z "$s" ]]; then
    echo 0; return
  fi
  # Count comma separated items
  local tmp="${s//,/ }"
  local c=0
  for _ in $tmp; do
    c=$((c+1))
  done
  echo "$c"
}

count_judges() {
  local specs="$1"
  local tmp="${specs//,/ }"
  local c=0
  for spec in $tmp; do
    spec="$(echo "$spec" | tr -d '[:space:]')"
    [[ -z "$spec" ]] && continue
    c=$((c+1))
  done
  echo "$c"
}

OPENAI_N="$(count_list_items "$GEN_OPENAI_MODELS")"
OLLAMA_N="$(count_list_items "$GEN_OLLAMA_MODELS")"
GROQ_N="$(count_list_items "$GEN_GROQ_MODELS")"
JUDGE_N="$(count_judges "$JUDGE_SPECS")"

# total “units”: each model generates 1 answers file, then each judge produces 1 judged file.
# So per model: (1 gen) + (JUDGE_N judge)
TOTAL_TASKS=$(( (OPENAI_N + OLLAMA_N + GROQ_N) * (1 + JUDGE_N) ))
_fmt_hms() {
  local sec="$1"
  sec=$(( sec < 0 ? 0 : sec ))
  local h=$(( sec / 3600 ))
  local m=$(( (sec % 3600) / 60 ))
  local s=$(( sec % 60 ))
  if [[ "$h" -gt 0 ]]; then
    printf "%dh%02dm%02ds" "$h" "$m" "$s"
  elif [[ "$m" -gt 0 ]]; then
    printf "%dm%02ds" "$m" "$s"
  else
    printf "%ds" "$s"
  fi
}

print_progress() {
  local msg="$1"

  local now_ts elapsed avg eta finish_ts pct
  now_ts="$(date +%s)"
  elapsed=$(( now_ts - START_TS ))

  if [[ "$TOTAL_TASKS" -gt 0 ]]; then
    pct="$(python - <<PY
done=${DONE_TASKS}
total=${TOTAL_TASKS}
print(f"{(done/total*100):.1f}")
PY
)"
  else
    pct="0.0"
  fi

  if [[ "$DONE_TASKS" -gt 0 ]]; then
    avg=$(( elapsed / DONE_TASKS ))
    eta=$(( avg * (TOTAL_TASKS - DONE_TASKS) ))
  else
    avg=0
    eta=0
  fi

  finish_ts=$(( now_ts + eta ))
  # macOS 的 date 格式略不同，用 -r 更通用
  local finish_str
  finish_str="$(date -r "$finish_ts" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date "+%Y-%m-%d %H:%M:%S")"

  echo "[PROGRESS] ${DONE_TASKS}/${TOTAL_TASKS} (${pct}%) | elapsed=$(_fmt_hms "$elapsed") avg/task=$(_fmt_hms "$avg") ETA=$(_fmt_hms "$eta") finish~${finish_str} | ${msg}"
}

bump_progress() {
  DONE_TASKS=$((DONE_TASKS + 1))
  print_progress "$1"
}



# -----------------------------
# Helpers: skip-if-exists
# -----------------------------
maybe_skip() {
  local path="$1"
  local what="$2"
  if [[ "$SKIP_EXISTING" == "1" && -f "$path" ]]; then
    echo "[SKIP] exists: $path (${what})"
    bump_progress "skip ${what}: $(basename "$path")"
    return 0
  fi
  return 1
}

# -----------------------------
# Generation helpers (no empty argv bug)
# -----------------------------
run_generation_openai() {
  local full="$1"
  local out="$2"
  if maybe_skip "$out" "gen openai:$full"; then
    return
  fi
  echo "[GEN] openai:$full -> $out"
  if [[ -n "${MAX_ITEMS:-}" ]]; then
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend openai \
      --openai-model "$full" \
      --max-items "$MAX_ITEMS" \
      --output "$out"
  else
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend openai \
      --openai-model "$full" \
      --output "$out"
  fi
  bump_progress "gen openai:$full"
}

run_generation_ollama() {
  local key="$1"
  local out="$2"
  if maybe_skip "$out" "gen ollama:$key"; then
    return
  fi
  echo "[GEN] ollama:$key -> $out"
  if [[ -n "${MAX_ITEMS:-}" ]]; then
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend ollama \
      --ollama-model-key "$key" \
      --max-items "$MAX_ITEMS" \
      --output "$out"
  else
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend ollama \
      --ollama-model-key "$key" \
      --output "$out"
  fi
  bump_progress "gen ollama:$key"
}

run_generation_groq() {
  local key="$1"
  local out="$2"
  if maybe_skip "$out" "gen groq:$key"; then
    return
  fi
  echo "[GEN] groq:$key -> $out"
  if [[ -n "${MAX_ITEMS:-}" ]]; then
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend groq \
      --groq-model-key "$key" \
      --max-items "$MAX_ITEMS" \
      --output "$out"
  else
    python -m src.run_generation \
      --scenarios "$SCENARIOS_FILE" \
      --backend groq \
      --groq-model-key "$key" \
      --output "$out"
  fi
  bump_progress "gen groq:$key"
}

# -----------------------------
# Judge runner (skip-if-exists, no empty argv)
# -----------------------------
run_one_judge() {
  local answers_path="$1"
  local backend="$2"
  local model_part="$3"

  if [[ "$backend" == "local" ]]; then
    backend="ollama"
  fi

  if [[ "$backend" == "openai" ]]; then
    local openai_model
    openai_model="$(resolve_openai_model "$model_part")"
    local out="$JUDGE_OUT_DIR/$(basename "$answers_path" .jsonl)_judged_openai_${openai_model//\//_}.jsonl"
    if maybe_skip "$out" "judge openai:$openai_model"; then
      return
    fi
    echo "[JUDGE] openai:$openai_model -> $out"
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
    bump_progress "judge openai:$openai_model"
    return
  fi

  if [[ "$backend" == "ollama" ]]; then
    local key
    key="$(normalize_ollama_key_or_name "$model_part")"
    local out="$JUDGE_OUT_DIR/$(basename "$answers_path" .jsonl)_judged_ollama_${key}.jsonl"
    if maybe_skip "$out" "judge ollama:$key"; then
      return
    fi
    echo "[JUDGE] ollama:$key -> $out"
    if [[ -n "${MAX_ITEMS:-}" ]]; then
      python -m src.run_judging \
        --answers "$answers_path" \
        --backend ollama \
        --ollama-model-key "$key" \
        --n-repeats "$N_REPEATS" \
        --max-tokens "$JUDGE_MAX_TOKENS" \
        --max-items "$MAX_ITEMS" \
        --output "$out"
    else
      python -m src.run_judging \
        --answers "$answers_path" \
        --backend ollama \
        --ollama-model-key "$key" \
        --n-repeats "$N_REPEATS" \
        --max-tokens "$JUDGE_MAX_TOKENS" \
        --output "$out"
    fi
    bump_progress "judge ollama:$key"
    return
  fi

  if [[ "$backend" == "groq" ]]; then
    local key
    key="$(normalize_groq_key_or_name "$model_part")"
    local out="$JUDGE_OUT_DIR/$(basename "$answers_path" .jsonl)_judged_groq_${key}.jsonl"
    if maybe_skip "$out" "judge groq:$key"; then
      return
    fi
    echo "[JUDGE] groq:$key -> $out"
    if [[ -n "${MAX_ITEMS:-}" ]]; then
      python -m src.run_judging \
        --answers "$answers_path" \
        --backend groq \
        --groq-model-key "$key" \
        --n-repeats "$N_REPEATS" \
        --max-tokens "$JUDGE_MAX_TOKENS" \
        --max-items "$MAX_ITEMS" \
        --output "$out"
    else
      python -m src.run_judging \
        --answers "$answers_path" \
        --backend groq \
        --groq-model-key "$key" \
        --n-repeats "$N_REPEATS" \
        --max-tokens "$JUDGE_MAX_TOKENS" \
        --output "$out"
    fi
    bump_progress "judge groq:$key"
    return
  fi

  echo "[WARN] Unknown judge backend '$backend' (skipped)"
}

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
    run_one_judge "$answers_path" "$backend" "$model_part"
  done
}

echo "============================================================"
echo "[INFO] LANGUAGE_MODE=$LANGUAGE_MODE"
echo "[INFO] SCENARIOS_FILE=$SCENARIOS_FILE"
echo "[INFO] MAX_ITEMS=${MAX_ITEMS:-ALL} N_REPEATS=$N_REPEATS JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
echo "[INFO] SKIP_EXISTING=$SKIP_EXISTING"
echo "[INFO] GEN_OPENAI_MODELS=$GEN_OPENAI_MODELS"
echo "[INFO] GEN_OLLAMA_MODELS=$GEN_OLLAMA_MODELS"
echo "[INFO] GEN_GROQ_MODELS=$GEN_GROQ_MODELS"
echo "[INFO] JUDGE_SPECS=$JUDGE_SPECS"
echo "[INFO] TOTAL_TASKS=$TOTAL_TASKS (per model: 1 gen + $JUDGE_N judge)"
echo "============================================================"

print_progress "start"

# -----------------------------
# Generation loops
# -----------------------------
low() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]'; }

if [[ "$(low "$GEN_OPENAI_MODELS")" != "none" ]]; then
  IFS=',' read -r -a OPENAI_LIST <<< "$GEN_OPENAI_MODELS"
  for m in "${OPENAI_LIST[@]}"; do
    m="$(echo "$m" | tr -d '[:space:]')"
    [[ -z "$m" ]] && continue
    full="$(resolve_openai_model "$m")"
    tag="openai_${full//\//_}"
    out="$MODEL_OUT_DIR/${LANGUAGE_MODE}_${tag}.jsonl"
    echo "------------------------------------------------------------"
    run_generation_openai "$full" "$out"
    run_judges_for_answers "$out" "$JUDGE_SPECS"
  done
else
  echo "[INFO] GEN_OPENAI_MODELS=none -> skipping OpenAI generation."
fi

if [[ "$(low "$GEN_OLLAMA_MODELS")" != "none" ]]; then
  IFS=',' read -r -a OLLAMA_LIST <<< "$GEN_OLLAMA_MODELS"
  for k in "${OLLAMA_LIST[@]}"; do
    k="$(echo "$k" | tr -d '[:space:]')"
    [[ -z "$k" ]] && continue
    key="$(normalize_ollama_key_or_name "$k")"
    tag="ollama_${key}"
    out="$MODEL_OUT_DIR/${LANGUAGE_MODE}_${tag}.jsonl"
    echo "------------------------------------------------------------"
    run_generation_ollama "$key" "$out"
    run_judges_for_answers "$out" "$JUDGE_SPECS"
  done
else
  echo "[INFO] GEN_OLLAMA_MODELS=none -> skipping Ollama generation."
fi

if [[ "$(low "$GEN_GROQ_MODELS")" != "none" ]]; then
  IFS=',' read -r -a GROQ_LIST <<< "$GEN_GROQ_MODELS"
  for k in "${GROQ_LIST[@]}"; do
    k="$(echo "$k" | tr -d '[:space:]')"
    [[ -z "$k" ]] && continue
    key="$(normalize_groq_key_or_name "$k")"
    tag="groq_${key}"
    out="$MODEL_OUT_DIR/${LANGUAGE_MODE}_${tag}.jsonl"
    echo "------------------------------------------------------------"
    run_generation_groq "$key" "$out"
    run_judges_for_answers "$out" "$JUDGE_SPECS"
  done
else
  echo "[INFO] GEN_GROQ_MODELS=none -> skipping Groq generation."
fi

print_progress "done"
echo "============================================================"
echo "[ALL DONE] Generation + Judging finished."
echo "Model outputs: $MODEL_OUT_DIR"
echo "Judge outputs: $JUDGE_OUT_DIR"
echo "============================================================"
