# parenting-advice-llm-evals

一个面向研究与工程的 LLM 评测系统，用于在真实亲子沟通场景中系统评估模型的育儿建议能力，支持英文与中文双语。

本仓库是一个关于评估大语言模型支持数字育儿能力的研究项目实现代码。

与只输出平均分的常规基准不同，本项目重点强调：

* 在统一流程下进行**多模型、跨厂商**评测
* 支持 **`en` / `zh` 双语与跨语言**分析
* 显式记录**工程失败**（空输出、截断、API 错误、重试）
* 流程**可诊断、可修复**，而非一次性跑分
* 支持生成**面向决策的长篇报告**（`.md` + `.docx`）

**数据来源致谢。** 种子场景集改编自 [ParentBench 项目](https://parentbench.azurewebsites.net)（由 NGO Early Ideas 开发）。本仓库在获得授权的前提下使用 ParentBench 场景作为研究数据，但本仓库的评测流水线是独立项目。

英文版请见 [README.md](README.md)。

---

## 1. 端到端流程概览

```
Scenario 构建
   ↓
Generation（多模型生成，含重试/截断处理）
   ↓
Quality Control（完整性扫描 + 定向重生成）
   ↓
Judging（单/多 Judge 结构化打分）
   ↓
Aggregation（CSV 导出 + 合并总表）
   ↓
Analysis（语言内 + 跨语言诊断）
   ↓
Auto Report（Vision 证据 → Writer 综合写作）
```

---

## 2. 仓库结构

```
parenting-advice-llm-evals/
├── data/
│   ├── scenarios/                  # scenario 定义与语言视图        [不在 repo]
│   ├── model_outputs/              # 生成结果（jsonl）              [不在 repo]
│   └── judge_outputs/              # 评分结果（jsonl）              [不在 repo]
│
├── results/                        # 图表/诊断/报告                 [不在 repo]
│
├── scripts/                        # 编排与工具脚本
├── src/                            # 核心实现
├── answer_repair/                  # 定向重生成 + 重判
│                                   #   （scripts/ + lib/ 在 repo；workspace/ 不在）
├── redesign/                       # 通用绘图辅助代码
└── README.md
```

本仓库**只包含框架与运作代码**。场景数据、生成/评分输出、分析产物、图表
中间件、以及 workspace 临时产物均不在版本控制中。获取或重建方式见第 3 节。

---

## 3. 获取 Scenario 数据

本仓库**不打包**场景数据。要端到端运行流水线，需要把场景文件放到
`data/scenarios/views/{en,zh}/parentbench_v0_{en,zh}.jsonl`（文件名里保留的
`parentbench_v0` 前缀只是数据版本号，与本仓库身份无关，是为了兼容代码里的
默认路径）。

本研究项目使用的种子场景集改编自 [ParentBench 项目](https://parentbench.azurewebsites.net)
（由 NGO Early Ideas 开发），仅在研究范围内得到使用授权。如需复现或扩展，
请直接联系 ParentBench 项目以获得相应授权。

如果你有自己的育儿场景 Excel 表，`scripts/convert_scenarios_from_excel.py`
可以把它转换成本仓库期望的 JSONL 视图格式。

---

## 4. 快速开始（最小可复现）

### 4.1 环境准备

推荐 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

若 `requirements.txt` 在你的环境中不可直接复用，可安装最小运行依赖：

```bash
pip install openai groq requests python-dotenv pandas numpy matplotlib seaborn scikit-learn tqdm openpyxl python-docx
```

### 4.2 配置 `.env`

在项目根目录创建并填写 `.env`（该文件已在 `.gitignore` 中，绝不要提交）：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_ALWAYS_SEND_KEY=0

GROQ_API_KEY=
GROQ_BASE_URL=
```

快速检查环境与模型：

```bash
python -m src.list_supported_models
```

### 4.3 生成 → 评分 → 合并 → 分析

1) 生成回答（示例：OpenAI `gpt-5.2`，英文场景）：

```bash
python -m src.run_generation \
  --scenarios data/scenarios/views/en/parentbench_v0_en.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --output data/model_outputs/en_openai_gpt-5.2.jsonl
```

2) 对生成结果打分（示例：OpenAI Judge `gpt-5.2`，重复 3 次）：

```bash
python -m src.run_judging \
  --answers data/model_outputs/en_openai_gpt-5.2.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --n-repeats 3 \
  --max-tokens 1024 \
  --output data/judge_outputs/en_openai_gpt-5.2_judged_openai_gpt-5.2.jsonl
```

3) 导出并合并评分：

```bash
python -m src.analysis.collect_and_merge \
  --judge-dir data/judge_outputs \
  --scores-dir results/scores \
  --merged-output results/merged/all_judge_scores.csv
```

4) 执行基础分析：

```bash
python -m src.analysis.analyze_scores \
  --input results/merged/all_judge_scores.csv \
  --out-dir results/analysis \
  --language all
```

---

## 5. Scenario 构建与语言视图

### 5.1 数据布局

当前语言视图分离存储：

* `data/scenarios/views/en/parentbench_v0_en.jsonl`
* `data/scenarios/views/zh/parentbench_v0_zh.jsonl`

每条记录均使用稳定 `scenario_uid`。

### 5.2 构建 / 转换 / 翻译

常用脚本：

```bash
python scripts/convert_scenarios_from_excel.py
python scripts/translate_scenarios_to_zh.py --backend openai --model gpt-4o-mini
python scripts/generate_zh_scenarios.py --theme sleep --age-group toddler --difficulty moderate --n 20
python scripts/build_scenario_views.py
```

---

## 6. 生成流程（Generation）

入口：`src/run_generation.py`

核心能力：

* 按语言构建 prompt
* 清洗推理痕迹（`answer_raw` 与 `answer` 分离）
* 截断可疑检测
* 截断后自动扩 token 重试
* 记录逐条元数据（`finish_reason`、`api_error`、`retry_used`）

示例（Ollama 注册表 key）：

```bash
python -m src.run_generation \
  --scenarios data/scenarios/views/zh/parentbench_v0_zh.jsonl \
  --backend ollama \
  --ollama-model-key deepseek_v3 \
  --max-items 20 \
  --output data/model_outputs/zh_ollama_deepseek_v3.jsonl
```

---

## 7. 质量控制与修复（Quality Control）

### 7.1 完整性扫描

```bash
python scripts/check_generation_completeness.py \
  --dir data/model_outputs \
  --out results/analysis/generation_completeness.csv
```

### 7.2 定向重生成

仅重跑问题样本（`empty` / `trunc` / `both`），且不覆盖原始文件：

```bash
python scripts/retry_bad_generations.py \
  --scenarios data/scenarios/views/zh/parentbench_v0_zh.jsonl \
  --inputs data/model_outputs \
  --pattern "zh_*.jsonl" \
  --retry-mode both \
  --attempts 15 \
  --suffix _retried
```

---

## 8. 评测流程（Judging）

入口：`src/run_judging.py`

要点：

* 严格 JSON rubric 评分
* `n_repeats` 支持 Judge 方差估计
* 对不规范模型输出进行鲁棒 JSON 解析
* 输出结构包含 `raw_judge_runs`、`api_errors` 与聚合后分数

对已有回答批量评分：

```bash
bash scripts/judge_existing_answers.sh "data/model_outputs/en_openai_gpt-5.2.jsonl"
```

---

## 9. 一键多模型运行

### 9.1 主编排脚本（生成 + 评分）

```bash
LANGUAGE_MODE=en \
MAX_ITEMS=20 \
GEN_OPENAI_MODELS="gpt_5_2,gpt_4o_mini" \
GEN_OLLAMA_MODELS="deepseek_v3,qwen3_8b" \
GEN_GROQ_MODELS=none \
JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3" \
N_REPEATS=3 \
bash scripts/run_multi_generation_and_judge.sh
```

### 9.2 EN + ZH 全流程

```bash
SKIP_EXISTING=1 \
N_REPEATS=3 \
bash scripts/run_full_en_zh.sh
```

---

## 10. 分析模块

基础分析：

```bash
python -m src.analysis.analyze_scores \
  --input results/merged/all_judge_scores.csv \
  --language all
```

分组能力报告：

```bash
python -m src.analysis.basic_grouped_capability_report \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/basic_capability_report
```

Rubric 分组验证：

```bash
python -m src.analysis.validate_rubric_split_anysize_ranked \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/rubric_split_anysize_ranked
```

高价值场景挖掘：

```bash
python -m src.analysis.scenario_value_mining \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/scenario_value_mining
```

模型总体能力与结构拆分：

```bash
python -m src.analysis.model_overall_competence_and_structure \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/model_overall_competence
```

---

## 11. 双维度 Parenting Style 流水线（R, D）

该项目包含「风格解释层」（非质量评价）：

* `R` = responsiveness，范围 `[0, 1]`
* `D` = demandingness，范围 `[0, 1]`
* 派生四类风格概率：
  - authoritative = `R * D`
  - permissive = `R * (1-D)`
  - authoritarian = `(1-R) * D`
  - neglectful = `(1-R) * (1-D)`

完整运行：

```bash
python -m src.run_style_judging \
  --model_outputs_dir data/model_outputs \
  --output_dir data/judge_outputs/style_2d \
  --judge_model gpt-5.2 \
  --judge_backend openai

python -m src.analysis.extract_style_2d \
  --input-dir data/judge_outputs/style_2d \
  --language all \
  --out-csv results/analysis/style_2d_tables/style_2d_rows.csv \
  --out-model-csv results/analysis/style_2d_tables/style_2d_model_summary.csv

python -m src.analysis.analyze_style_2d \
  --rows-csv results/analysis/style_2d_tables/style_2d_rows.csv \
  --out-dir results/analysis \
  --language all
```

---

## 12. 自动长报告生成

报告脚本（`scripts/generate_llm_report_with_vision.py`）分两阶段：

1. **Vision 阶段**：从图像提取证据
2. **Writer 阶段**：融合表格 + 文本 + 图像证据生成长报告

```bash
python scripts/generate_llm_report_with_vision.py \
  --project-root . \
  --analysis-dir results/analysis \
  --include-merged \
  --vision-model gpt-4o-mini \
  --report-model gpt-5-nano \
  --max-images 200
```

Markdown 转 DOCX：

```bash
python scripts/markdown_to_docx.py \
  --md results/analysis/report/<report>.md \
  --out results/analysis/report/<report>.docx
```

---

## 13. 输出产物速查

* 生成结果：`data/model_outputs/*.jsonl`
* Judge 结果：`data/judge_outputs/*.jsonl`
* 按文件评分 CSV：`results/scores/*.csv`
* 合并总表：`results/merged/all_judge_scores.csv`
* 基础分析图：`results/analysis/{en|zh|all}/...`
* Style-2D 表：`results/analysis/style_2d_tables/*.csv`
* 自动报告：`results/analysis/report/*`

以上目录都已在 `.gitignore` 中，新克隆后是空的，需要本地运行流水线产出。

---

## 14. 当前代码库注意事项

* Groq 在模型注册与脚本编排中可配置，但在当前分支正式运行前仍建议先做端到端验证。
* `scripts/judge_existing_answers.sh` 含历史兼容分支，推荐优先使用 `scripts/run_multi_generation_and_judge.sh`。
* CSV 物理行数可能大于逻辑记录数（`comment` 字段可能含换行），统计请以 DataFrame 行数为准。
* 场景文件名里的历史前缀 `parentbench_v0` 仅是数据版本标识，与本仓库身份无关，保留只是为了与代码默认路径一致。
* `redesign/chart_utils.py` 作为通用绘图辅助代码保留，并不是自包含的：它依赖的 `chart_config.py`、各章渲染器、图表输出目录都没有打包进 repo。

---

## 15. 推荐运行实践

* 长任务放在 `tmux` 中执行。
* 设置 `SKIP_EXISTING=1` 便于断点续跑。
* 先用 `MAX_ITEMS` 做小样本冒烟测试，再跑全量。
* 保留所有中间产物，不覆盖原始输出。
* 报告中区分「模型能力结论」和「工程稳定性结论」。

---

## License

仅用于研究与内部评测。如有基于本仓库的工作，请引用本仓库。
