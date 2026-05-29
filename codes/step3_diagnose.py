"""
DiagSQL Step 3: SQL Diagnosis Evaluation

使用训练好的 DiagSQL 模型对 NL2SQL-Bugs 数据集进行 SQL 诊断测试。

Usage:
    python step3_diagnose.py --config configs/config_test.yaml
    python step3_diagnose.py --config configs/config_full.yaml
    python step3_diagnose.py --config configs/config_test.yaml --model_path /path/to/model
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DiagSQL Step 3: SQL Diagnosis")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.chdir(str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent))

    output_dir = Path(cfg["output_dir"])
    step3_cfg = cfg["step3"]

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        info_path = output_dir / "model_paths.json"
        if info_path.exists():
            with open(info_path) as f:
                model_path = json.load(f)["grpo_model"]
        else:
            raise FileNotFoundError(f"No model_paths.json found at {info_path}. Run step2 first or use --model_path.")

    logger.info("=" * 60)
    logger.info("  Step 3: SQL Diagnosis Evaluation")
    logger.info(f"  Model: {model_path}")
    logger.info("=" * 60)

    from pipeline_diagsql import DiagSQLValidator, get_database_schema, execute_sql, format_execution_result

    # Load test data
    with open(cfg["nl2sql_bugs_path"], 'r') as f:
        nl2sql_data = json.load(f)

    num_samples = step3_cfg["num_test_samples"]
    if num_samples > 0:
        test_data = nl2sql_data[:num_samples]
    else:
        test_data = nl2sql_data
    logger.info(f"  Test samples: {len(test_data)}")

    # Initialize validator
    validator = DiagSQLValidator(
        model_path=model_path,
        tensor_parallel_size=step3_cfg["tensor_parallel_size"],
        gpu_memory_utilization=step3_cfg["gpu_memory_utilization"],
        max_model_len=step3_cfg["max_model_len"],
    )

    # Prepare inputs
    bird_db_path = cfg["bird_db_path"]
    diag_inputs = []
    for item in test_data:
        schema = get_database_schema(item["db_id"], bird_db_path)
        exec_result = execute_sql(item["db_id"], item.get("sql", ""), bird_db_path)
        exec_str = format_execution_result(exec_result)
        diag_inputs.append({
            "question": item["question"],
            "evidence": item.get("evidence", ""),
            "schema": schema,
            "sql": item.get("sql", ""),
            "execution_result": exec_str,
        })

    # Batch diagnosis
    logger.info(f"  Diagnosing {len(diag_inputs)} SQL queries...")
    diagnoses = validator.diagnose_batch(diag_inputs)

    # Evaluate
    correct = 0
    results = []
    for item, diag in zip(test_data, diagnoses):
        gt_label = "right" if item["label"] else "wrong"
        pred_label = diag.get("label", "unknown")
        is_correct = (pred_label == gt_label)
        correct += int(is_correct)
        results.append({
            "question": item["question"],
            "db_id": item["db_id"],
            "gt_label": gt_label,
            "pred_label": pred_label,
            "correct": is_correct,
            "diagnosis": diag,
        })

    accuracy = correct / len(test_data) * 100
    logger.info(f"  Diagnosis accuracy: {correct}/{len(test_data)} = {accuracy:.1f}%")

    # Save
    out_path = output_dir / "step3_diagnosis_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            "model_path": model_path,
            "total": len(test_data),
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"  Results saved to {out_path}")
    logger.info(f"✓ Step 3 完成 (accuracy={accuracy:.1f}%)")


if __name__ == "__main__":
    main()
