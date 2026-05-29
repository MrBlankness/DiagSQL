# DiagSQL: A Diagnostic Validator for Text-to-SQL

> Closed-Loop Text-to-SQL: **GPT-4o generates SQL → DiagSQL diagnoses → GPT-4o refines → repeat for K rounds**
>
> A fine-grained SQL error diagnostic model trained with **FTRA** (Fine-grained Token-level Reward Allocation) + **CORS** (Co-occurrence-aware Reward Shaping).

---

## Repository Structure

```
DiagSQL/
├── codes/          # All source code (training, evaluation, pipeline)
│   ├── README.md   # Code usage documentation (detailed step1~step4 instructions)
│   ├── configs/    # YAML configurations
│   └── ...
└── dataset/        # Data directory (datasets are NOT included; see below for download instructions)
```

For code usage details, please refer to [`codes/README.md`](codes/README.md).

---

## Dataset Preparation

Due to their large size and the fact that they are publicly available third-party datasets, this repository **does not include any data under `dataset/`**. Please follow the steps below to download and place them into the corresponding directories.

The final directory layout should look like:

```
DiagSQL/
└── dataset/
    ├── NL2SQLBugs/
    │   ├── NL2SQL-Bugs-with-evidence.json
    │   ├── NL2SQL-Bugs-with-evidence-enriched.json
    │   ├── generated_error_data_2k.json
    │   ├── cooccurrence_matrix.json
    │   └── cooccurrence_matrix.npy
    ├── BIRD/
    │   └── dev_20240627/
    │       ├── dev.json
    │       ├── dev.sql
    │       ├── dev_tables.json
    │       └── dev_databases/
    └── spider_data/
        ├── dev.json
        ├── dev_gold.sql
        ├── tables.json
        ├── test.json
        ├── test_gold.sql
        ├── test_tables.json
        ├── train_gold.sql
        ├── train_others.json
        ├── train_spider.json
        ├── database/
        └── test_database/
```

### 1. NL2SQL-Bugs

- Paper: *NL2SQL-Bugs: A Benchmark for Detecting Semantic Errors in NL2SQL Translation*
- Project page: <https://nl2sql-bugs.github.io/>
- GitHub: <https://github.com/HKUSTDial/NL2SQL-Bugs>
- Files to download:
  - `NL2SQL-Bugs-with-evidence.json` (~1.17 MB, 2,018 diagnostic samples)
  - `NL2SQL-Bugs-with-evidence-enriched.json` (enriched version)
- Place under: `dataset/NL2SQLBugs/`

> `generated_error_data_2k.json`, `cooccurrence_matrix.json`, and `cooccurrence_matrix.npy` are produced by the scripts in this project (see Step 1 in `codes/README.md`).

### 2. BIRD Benchmark (Dev)

- Paper: *BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQLs*
- Project page: <https://bird-bench.github.io/>
- Download: <https://bird-bench.oss-cn-beijing.aliyuncs.com/dev_20240627.zip>
- After unzipping, place the entire `dev_20240627/` directory under: `dataset/BIRD/dev_20240627/`

### 3. Spider

- Paper: *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*
- Project page: <https://yale-lily.github.io/spider>
- GitHub: <https://github.com/taoyds/spider>
- Download (official Google Drive link): <https://drive.google.com/uc?export=download&id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J>
- After unzipping you will get `spider_data/`; place the entire directory under: `dataset/spider_data/`

### 4. Base Model

This project uses **Qwen3-8B** as the base model by default:

- HuggingFace: <https://huggingface.co/Qwen/Qwen3-8B>
- Download to any path, then update the `base_model_path` field in `codes/configs/config_full.yaml`.

---

## Quick Start

```bash
# 1. Enter the project directory
cd DiagSQL

# 2. Download datasets into dataset/ as instructed above

# 3. Set up the environment
conda create -n nl2sql_bugs python=3.10 -y
conda activate nl2sql_bugs
pip install torch transformers accelerate trl peft datasets vllm tensorboard pyyaml requests tqdm

# 4. Run the 4-step pipeline
cd codes
python step1_build_data.py --config configs/config_full.yaml
python step2_train.py      --config configs/config_full.yaml
python step3_diagnose.py   --config configs/config_full.yaml
python step4_pipeline.py   --config configs/config_full.yaml
```

For detailed parameter descriptions, see [`codes/README.md`](codes/README.md).

---

## Hardware Requirements

- 4× NVIDIA H20 (96 GB) or equivalent GPUs
- SFT: `device_map="auto"` + LoRA
- GRPO: `device_map="auto"` + gradient checkpointing
- Inference: vLLM with `tensor_parallel_size=4`

---

## Acknowledgements

This project is built upon the following public datasets and open-source models. We thank the authors for their work:

- [NL2SQL-Bugs](https://nl2sql-bugs.github.io/)
- [BIRD-Bench](https://bird-bench.github.io/)
- [Spider](https://yale-lily.github.io/spider)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
