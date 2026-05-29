# DiagSQL: A Diagnostic Validator for Text-to-SQL

## 概述

DiagSQL 训练 SQL 诊断验证器，实现 Closed-Loop Text-to-SQL：

```
GPT-4o 生成 SQL → DiagSQL 诊断 → GPT-4o 修复 → 循环 K 轮
```

核心算法：
- **FTRA** (Fine-grained Token-level Reward Allocation)：将奖励精确分配到 JSON 响应的各个 token span
- **CORS** (Co-occurrence-aware Reward Shaping)：利用预计算的错误共现矩阵调整奖励

## 环境准备

```bash
conda activate nl2sql_bugs
# 依赖: torch, transformers, accelerate, trl, peft, datasets, vllm, tensorboard, pyyaml, requests, tqdm
```

## 快速开始

整个流程分为 4 个独立的 Step，通过**配置文件**控制运行规模：
- `configs/config_full.yaml`：完整训练 + 全量评估

---


使用 `config_full.yaml` 进行正式训练：

```bash
cd DiagSQL/codes
conda activate nl2sql_bugs

# Step 1: 构造全量数据（SFT + GRPO 20000条）
python step1_build_data.py --config configs/config_full.yaml

# Step 2: 完整训练（SFT + GRPO
python step2_train.py --config configs/config_full.yaml

# Step 3: 全量诊断测试（2018条 NL2SQLBugs）
python step3_diagnose.py --config configs/config_full.yaml

# Step 4: 完整管线评估（BIRD 1534条 + Spider 1034条，K=3）
python step4_pipeline.py --config configs/config_full.yaml
```

### 跳过 SFT，使用已有 checkpoint

如果已有 SFT checkpoint，可以跳过 SFT 直接训练 GRPO：

```bash
python step2_train.py --config configs/config_full.yaml \
    --skip_sft \
    --sft_model_path /path/to/diagsql_checkpoint_sft
```

### 使用指定模型运行 Step 3/4

```bash
# 用已训练好的模型进行诊断
python step3_diagnose.py --config configs/config_full.yaml \
    --model_path /path/to/diagsql_checkpoint

# 用已训练好的模型跑完整管线
python step4_pipeline.py --config configs/config_full.yaml \
    --model_path /path/to/diagsql_checkpoint
```


## 目录结构

```
codes/
├── 流程脚本（按步骤拆分）
│   ├── step1_build_data.py       # 数据构造
│   ├── step2_train.py            # 模型训练（SFT + GRPO）
│   ├── step3_diagnose.py         # SQL 诊断评估
│   └── step4_pipeline.py         # 完整管线（生成→诊断→修复→评估）
│
├── 配置文件
│   └── configs/
│       └── config_full.yaml      # 完整配置（全量训练评估）
│
├── 核心算法
│   ├── train_diagsql.py          # GRPO RL 训练引擎
│   ├── grpo.py                   # GRPO rollout + policy update
│   ├── ftra.py                   # Fine-grained Token-level Reward Allocation
│   ├── cors.py                   # Co-occurrence-aware Reward Shaping
│   ├── dataset.py                # 训练数据加载器
│   └── data_types.py             # 数据结构定义
│
├── 管线与评估
│   ├── pipeline_diagsql.py       # Closed-Loop 管线（GPT-4o + DiagSQL）
│   └── eval_bird.py              # BIRD/Spider 执行准确率评估
│
└── 工具
    ├── export_cooccurrence.py    # 预计算共现矩阵
    ├── analyze_error_causality.py # 错误共现分析
    └── generate_error_sql.py     # 生成错误 SQL
```

---

## 算法说明

### FTRA：将响应解析为 span，独立分配奖励

```
Response → {s_fmt, s_label, s_err^(1), ..., s_err^(K)}

r_fmt   → 均匀分配到所有 token
r_label → 分配到 "label":"wrong" 对应的 token
r_err^k → 分配到第 k 个 error_type 对象的 token（含 CORS 调整）
```

### CORS：基于共现矩阵的结构化奖励调整

```
r_err^(u) = r_base^(u) + λ · Σ_{v≠u} [M_{u,v} · b_v · b̂_v - |M_{u,v}| · (b_v - b̂_v)²]
```

- `M[u,v] = log2(Lift(u,v))`：预计算的错误对共现强度
- Associative Bonus：正确预测共现错误对 → 奖励
- Cascading Penalty：遗漏强关联错误 → 惩罚
- Illogical Combination：幻觉不相关错误 → 惩罚

---

## 数据集

| 数据集 | 路径 | 规模 |
|--------|------|------|
| NL2SQL-Bugs | `dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence.json` | 2018 条 |
| GRPO 训练数据 | `dataset/NL2SQLBugs/generated_error_data_2k.json` | 2000 条 |
| 共现矩阵 | `dataset/NL2SQLBugs/cooccurrence_matrix.json` | 466 条关系 |
| BIRD Dev | `dataset/BIRD/dev_20240627/dev.json` | 1534 条 |
| Spider Dev | `dataset/spider_data/dev.json` | 1034 条 |

## 硬件要求

- 4× NVIDIA H20 (96GB) / 同等规格 GPU
- SFT 训练：`device_map="auto"` + LoRA
- GRPO 训练：`device_map="auto"` + gradient checkpointing
- 推理：vLLM `tensor_parallel_size=4`
