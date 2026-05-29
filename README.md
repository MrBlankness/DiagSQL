# DiagSQL: A Diagnostic Validator for Text-to-SQL

> Closed-Loop Text-to-SQL：**GPT-4o 生成 SQL → DiagSQL 诊断 → GPT-4o 修复 → 循环 K 轮**
>
> 通过 **FTRA**（Fine-grained Token-level Reward Allocation）+ **CORS**（Co-occurrence-aware Reward Shaping）训练得到一个细粒度 SQL 错误诊断模型。

---

## 仓库结构

```
DiagSQL/
├── codes/          # 全部代码（训练、评估、管线）
│   ├── README.md   # 代码使用文档（详细的 step1~step4 说明）
│   ├── configs/    # YAML 配置
│   └── ...
└── dataset/        # 数据目录（本仓库不包含数据，需自行下载，见下文）
```

代码使用方式请参考 [`codes/README.md`](codes/README.md)。

---

## 数据集准备

由于体积过大且属于公开第三方数据集，本仓库**不包含 `dataset/` 中的数据**。请按以下步骤自行下载并放置到对应目录。

最终目录结构应为：

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

- 论文：*NL2SQL-Bugs: A Benchmark for Detecting Semantic Errors in NL2SQL Translation*
- 项目主页：<https://nl2sql-bugs.github.io/>
- GitHub：<https://github.com/HKUSTDial/NL2SQL-Bugs>
- 需要下载：
  - `NL2SQL-Bugs-with-evidence.json`（约 1.17 MB，2018 条诊断样本）
  - `NL2SQL-Bugs-with-evidence-enriched.json`（增强版本）
- 放置到：`dataset/NL2SQLBugs/`

> `generated_error_data_2k.json`、`cooccurrence_matrix.json`、`cooccurrence_matrix.npy` 由本项目脚本生成（详见 `codes/README.md` 中的 Step 1）。

### 2. BIRD Benchmark (Dev)

- 论文：*BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQLs*
- 项目主页：<https://bird-bench.github.io/>
- 下载链接：<https://bird-bench.oss-cn-beijing.aliyuncs.com/dev_20240627.zip>
- 解压后将整个 `dev_20240627/` 目录放置到：`dataset/BIRD/dev_20240627/`

### 3. Spider

- 论文：*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*
- 项目主页：<https://yale-lily.github.io/spider>
- GitHub：<https://github.com/taoyds/spider>
- 下载（Google Drive 官方链接）：<https://drive.google.com/uc?export=download&id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J>
- 解压得到 `spider_data/`，整个目录放置到：`dataset/spider_data/`

### 4. 基础模型

本项目默认使用 **Qwen3-8B** 作为基座模型：

- HuggingFace：<https://huggingface.co/Qwen/Qwen3-8B>
- 下载后放置到任意路径，并在 `codes/configs/config_full.yaml` 中修改 `base_model_path` 字段。

---

## 快速开始

```bash
# 1. 进入项目目录
cd DiagSQL

# 2. 按上文说明下载数据集到 dataset/ 目录

# 3. 准备环境
conda create -n nl2sql_bugs python=3.10 -y
conda activate nl2sql_bugs
pip install torch transformers accelerate trl peft datasets vllm tensorboard pyyaml requests tqdm

# 4. 运行 4 步流水线
cd codes
python step1_build_data.py --config configs/config_full.yaml
python step2_train.py      --config configs/config_full.yaml
python step3_diagnose.py   --config configs/config_full.yaml
python step4_pipeline.py   --config configs/config_full.yaml
```

详细参数说明请见 [`codes/README.md`](codes/README.md)。

---

## 硬件要求

- 4× NVIDIA H20 (96GB) 或同等规格 GPU
- SFT：`device_map="auto"` + LoRA
- GRPO：`device_map="auto"` + gradient checkpointing
- 推理：vLLM `tensor_parallel_size=4`

---

## 致谢

本项目基于以下公开数据集与开源模型构建，感谢相关作者的工作：

- [NL2SQL-Bugs](https://nl2sql-bugs.github.io/)
- [BIRD-Bench](https://bird-bench.github.io/)
- [Spider](https://yale-lily.github.io/spider)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
