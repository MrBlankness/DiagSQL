"""
DiagSQL Step 1: Data Construction

构造 SFT 数据和 GRPO 数据。

Usage:
    python step1_build_data.py --config configs/config_test.yaml
    python step1_build_data.py --config configs/config_full.yaml
"""
import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DiagSQL Step 1: Data Construction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    sft_dir = output_dir / "sft_data"
    sft_dir.mkdir(parents=True, exist_ok=True)

    sft_num_samples = cfg["step1"]["sft_num_samples"]
    grpo_num_entries = cfg["step1"]["grpo_num_entries"]

    logger.info("=" * 60)
    logger.info("  Step 1: Data Construction")
    logger.info(f"  SFT samples: {sft_num_samples}, GRPO entries: {grpo_num_entries}")
    logger.info("=" * 60)

    # ── Import pipeline utilities ──
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline_diagsql import get_database_schema, execute_sql, format_execution_result
    from dataset import SYSTEM_MESSAGE

    USER_TEMPLATE = (
        "Question: {question}\n"
        "Evidence: {evidence}\n"
        "Database Schema:\n{schema}\n"
        "Generated SQL: {sql}\n"
        "Execution Result: {execution_result}\n\n"
        "Please analyze whether this SQL query correctly answers the question based on the provided evidence, "
        "database schema, and execution result. "
        "If the SQL is incorrect, identify the specific error types."
    )

    # ═══ 1a: Construct SFT data ═══
    logger.info("[1a] Constructing SFT data from NL2SQL-Bugs...")
    with open(cfg["nl2sql_bugs_path"], 'r') as f:
        nl2sql_data = json.load(f)

    bird_db_path = cfg["bird_db_path"]
    sft_samples = []
    for item in nl2sql_data[:sft_num_samples]:
        question = item["question"]
        evidence = item.get("evidence", "")
        db_id = item["db_id"]
        sql = item.get("sql", item.get("SQL", ""))
        label = item.get("label", True)
        error_types = item.get("error_types", [])

        schema = get_database_schema(db_id, bird_db_path)
        exec_result = execute_sql(db_id, sql, bird_db_path)
        exec_str = format_execution_result(exec_result)

        user_msg = USER_TEMPLATE.format(
            question=question, evidence=evidence, schema=schema,
            sql=sql, execution_result=exec_str,
        )

        if label:
            assistant_content = json.dumps({"label": "right"})
        else:
            et_list = []
            for et in error_types:
                et_list.append({
                    "reasoning": f"The SQL has a {et.get('sub_error_type', 'error')} issue.",
                    "error_type": et.get("error_type", "Other Errors"),
                    "sub_error_type": et.get("sub_error_type", "Other"),
                })
            if not et_list:
                et_list = [{"reasoning": "SQL error.", "error_type": "Other Errors", "sub_error_type": "Other"}]
            assistant_content = json.dumps({"label": "wrong", "error_types": et_list})

        sft_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "nl2sql_bugs",
            "label": "right" if label else "wrong",
        })

    # Split train/val (90/10)
    split_idx = int(len(sft_samples) * 0.9)
    train_samples = sft_samples[:split_idx]
    val_samples = sft_samples[split_idx:]

    with open(sft_dir / "sft_train.jsonl", 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(sft_dir / "sft_val.jsonl", 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"  SFT data: {len(train_samples)} train + {len(val_samples)} val")
    logger.info(f"  Saved to {sft_dir}")

    # ═══ 1b: Construct GRPO data ═══
    logger.info("[1b] Constructing GRPO data from generated_error_data_2k...")
    with open(cfg["grpo_source_data_path"], 'r') as f:
        grpo_raw = json.load(f)

    grpo_subset = grpo_raw[:grpo_num_entries]
    grpo_output_path = output_dir / "grpo_data.json"
    with open(grpo_output_path, 'w') as f:
        json.dump(grpo_subset, f, ensure_ascii=False, indent=2)

    n_pos = len(grpo_subset)
    n_neg = sum(len(item.get("error_results", [])) for item in grpo_subset)
    logger.info(f"  GRPO data: {n_pos} positive + {n_neg} negative = {n_pos + n_neg} samples")
    logger.info(f"  Saved to {grpo_output_path}")

    logger.info("✓ Step 1 完成")


if __name__ == "__main__":
    main()
