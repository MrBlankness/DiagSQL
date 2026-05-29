# DiagSQL: A Diagnostic Validator for Text-to-SQL

## Overview

DiagSQL trains an SQL diagnostic validator to enable Closed-Loop Text-to-SQL:

```
GPT-4o generates SQL → DiagSQL diagnoses → GPT-4o refines → repeat for K rounds
```

Core algorithms:
- **FTRA** (Fine-grained Token-level Reward Allocation): precisely allocates rewards to each token span of the JSON response
- **CORS** (Co-occurrence-aware Reward Shaping): adjusts rewards using a precomputed error co-occurrence matrix

## Environment Setup

```bash
conda activate nl2sql_bugs
# Dependencies: torch, transformers, accelerate, trl, peft, datasets, vllm, tensorboard, pyyaml, requests, tqdm
```

## Quick Start

The full workflow consists of 4 independent steps, controlled by a **configuration file**:
- `configs/config_full.yaml`: full training + full evaluation

---

Use `config_full.yaml` for full-scale training:

```bash
cd DiagSQL/codes
conda activate nl2sql_bugs

# Step 1: Build the full training data (SFT + 20,000 GRPO samples)
python step1_build_data.py --config configs/config_full.yaml

# Step 2: Full training (SFT + GRPO)
python step2_train.py --config configs/config_full.yaml

# Step 3: Full diagnostic evaluation (2,018 NL2SQL-Bugs samples)
python step3_diagnose.py --config configs/config_full.yaml

# Step 4: Full pipeline evaluation (1,534 BIRD + 1,034 Spider, K=3)
python step4_pipeline.py --config configs/config_full.yaml
```

### Skip SFT and reuse an existing checkpoint

If you already have an SFT checkpoint, you can skip SFT and train GRPO directly:

```bash
python step2_train.py --config configs/config_full.yaml \
    --skip_sft \
    --sft_model_path /path/to/diagsql_checkpoint_sft
```

### Run Step 3/4 with a specified model

```bash
# Run diagnosis with a trained model
python step3_diagnose.py --config configs/config_full.yaml \
    --model_path /path/to/diagsql_checkpoint

# Run the full pipeline with a trained model
python step4_pipeline.py --config configs/config_full.yaml \
    --model_path /path/to/diagsql_checkpoint
```

## Directory Structure

```
codes/
├── Pipeline scripts (split by step)
│   ├── step1_build_data.py       # Data construction
│   ├── step2_train.py            # Model training (SFT + GRPO)
│   ├── step3_diagnose.py         # SQL diagnostic evaluation
│   └── step4_pipeline.py         # Full pipeline (generate → diagnose → refine → evaluate)
│
├── Configuration files
│   └── configs/
│       └── config_full.yaml      # Full configuration (full training & evaluation)
│
├── Core algorithms
│   ├── train_diagsql.py          # GRPO RL training engine
│   ├── grpo.py                   # GRPO rollout + policy update
│   ├── ftra.py                   # Fine-grained Token-level Reward Allocation
│   ├── cors.py                   # Co-occurrence-aware Reward Shaping
│   ├── dataset.py                # Training data loader
│   └── data_types.py             # Data structure definitions
│
├── Pipeline & evaluation
│   ├── pipeline_diagsql.py       # Closed-Loop pipeline (GPT-4o + DiagSQL)
│   └── eval_bird.py              # BIRD/Spider execution accuracy evaluation
│
└── Utilities
    ├── export_cooccurrence.py    # Precompute the co-occurrence matrix
    ├── analyze_error_causality.py # Error co-occurrence analysis
    └── generate_error_sql.py     # Generate erroneous SQL
```

---

## Algorithm Details

### FTRA: Parse the response into spans and allocate rewards independently

```
Response → {s_fmt, s_label, s_err^(1), ..., s_err^(K)}

r_fmt   → distributed uniformly over all tokens
r_label → assigned to tokens of "label":"wrong"
r_err^k → assigned to tokens of the k-th error_type object (with CORS adjustment)
```

### CORS: Structural reward shaping based on the co-occurrence matrix

```
r_err^(u) = r_base^(u) + λ · Σ_{v≠u} [M_{u,v} · b_v · b̂_v - |M_{u,v}| · (b_v - b̂_v)²]
```

- `M[u,v] = log2(Lift(u,v))`: precomputed co-occurrence strength of error pairs
- Associative Bonus: correctly predicting co-occurring error pairs → reward
- Cascading Penalty: missing strongly-associated errors → penalty
- Illogical Combination: hallucinating unrelated errors → penalty

---

## Datasets

| Dataset | Path | Size |
|---------|------|------|
| NL2SQL-Bugs | `dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence.json` | 2,018 samples |
| GRPO training data | `dataset/NL2SQLBugs/generated_error_data_2k.json` | 2,000 samples |
| Co-occurrence matrix | `dataset/NL2SQLBugs/cooccurrence_matrix.json` | 466 relations |
| BIRD Dev | `dataset/BIRD/dev_20240627/dev.json` | 1,534 samples |
| Spider Dev | `dataset/spider_data/dev.json` | 1,034 samples |

## Hardware Requirements

- 4× NVIDIA H20 (96 GB) or equivalent GPUs
- SFT training: `device_map="auto"` + LoRA
- GRPO training: `device_map="auto"` + gradient checkpointing
- Inference: vLLM with `tensor_parallel_size=4`
