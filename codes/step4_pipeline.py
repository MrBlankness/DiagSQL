"""
DiagSQL Step 4: Full Closed-Loop Pipeline

GPT-4o 生成 SQL → DiagSQL 诊断 → GPT-4o 修复 → 评估 EX
分别在 BIRD 和 Spider 上测试。

Usage:
    python step4_pipeline.py --config configs/config_test.yaml
    python step4_pipeline.py --config configs/config_full.yaml
    python step4_pipeline.py --config configs/config_test.yaml --model_path /path/to/model
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DiagSQL Step 4: Full Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.chdir(str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent))

    output_dir = Path(cfg["output_dir"])
    step4_cfg = cfg["step4"]

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        info_path = output_dir / "model_paths.json"
        if info_path.exists():
            with open(info_path) as f:
                model_path = json.load(f)["grpo_model"]
        else:
            raise FileNotFoundError(f"No model_paths.json found. Run step2 first or use --model_path.")

    logger.info("=" * 60)
    logger.info("  Step 4: Full Closed-Loop Pipeline")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  K={step4_cfg['max_refine_rounds']} refine rounds")
    logger.info("=" * 60)

    from pipeline_diagsql import (
        GPT4oClient, DiagSQLValidator, load_dataset,
        get_database_schema, generate_sql, execute_sql, format_execution_result,
        refine_sql, evaluate_execution_accuracy,
    )

    # Initialize
    gpt_client = GPT4oClient(model=step4_cfg["gpt_model"], temperature=step4_cfg["gpt_temperature"])
    validator = DiagSQLValidator(
        model_path=model_path,
        tensor_parallel_size=step4_cfg["tensor_parallel_size"],
        gpu_memory_utilization=step4_cfg["gpu_memory_utilization"],
        max_model_len=step4_cfg["max_model_len"],
    )

    max_refine_rounds = step4_cfg["max_refine_rounds"]

    # Test on both BIRD and Spider
    datasets_to_test = [
        ("bird", cfg["bird_dev_json"], cfg["bird_db_path"], step4_cfg["bird_num_samples"]),
        ("spider", cfg["spider_dev_json"], cfg["spider_db_path"], step4_cfg["spider_num_samples"]),
    ]

    all_summaries = {}

    for dataset_name, dev_json, db_path, num_samples in datasets_to_test:
        max_samples = num_samples if num_samples > 0 else None
        logger.info(f"\n{'='*60}")
        logger.info(f"  Testing {dataset_name.upper()} (samples={'ALL' if not max_samples else max_samples}, K={max_refine_rounds})")
        logger.info(f"{'='*60}")

        data = load_dataset(dataset_name, dev_json, db_path, max_samples=max_samples)

        # Phase 1: Generate SQL
        logger.info("  Phase 1: Generating SQL with GPT-4o...")
        results = []
        for item in tqdm(data, desc=f"  {dataset_name} SQL Gen"):
            schema = get_database_schema(item["db_id"], db_path)
            sql = generate_sql(gpt_client, item["question"], item["evidence"], schema)
            results.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "evidence": item["evidence"],
                "db_id": item["db_id"],
                "ground_truth_sql": item["ground_truth_sql"],
                "schema": schema,
                "initial_sql": sql,
                "current_sql": sql,
                "final_sql": sql,
            })

        # Phase 2: Iterative Diagnosis + Refinement
        for round_idx in range(max_refine_rounds):
            logger.info(f"  Phase 2: Refinement Round {round_idx+1}/{max_refine_rounds}")

            items_to_diagnose = [r for r in results if not r.get("diagnosed_right", False)]
            if not items_to_diagnose:
                logger.info("    All confirmed correct. Stopping early.")
                break

            diag_inputs = []
            for r in items_to_diagnose:
                exec_result = execute_sql(r["db_id"], r["current_sql"], db_path)
                exec_str = format_execution_result(exec_result)
                diag_inputs.append({
                    "question": r["question"],
                    "evidence": r["evidence"],
                    "schema": r["schema"],
                    "sql": r["current_sql"],
                    "execution_result": exec_str,
                })

            diagnoses = validator.diagnose_batch(diag_inputs)

            refine_count = 0
            for r, diag in zip(items_to_diagnose, diagnoses):
                if diag.get("label") == "right":
                    r["diagnosed_right"] = True
                    continue
                refine_count += 1
                new_sql = refine_sql(
                    gpt_client, r["question"], r["evidence"],
                    r["schema"], r["current_sql"], diag,
                )
                r["current_sql"] = new_sql
                r["final_sql"] = new_sql

            logger.info(f"    {refine_count} refined, {len(items_to_diagnose)-refine_count} correct")

        # Phase 3: Evaluate
        logger.info("  Phase 3: Evaluating execution accuracy...")
        initial_correct = 0
        final_correct = 0
        difficulty_stats = {}

        for r in results:
            init_acc = evaluate_execution_accuracy(r["initial_sql"], r["ground_truth_sql"], r["db_id"], db_path)
            final_acc = evaluate_execution_accuracy(r["final_sql"], r["ground_truth_sql"], r["db_id"], db_path)
            r["initial_correct"] = init_acc
            r["final_correct"] = final_acc
            initial_correct += init_acc
            final_correct += final_acc

        total = len(results)
        init_pct = initial_correct / total * 100 if total else 0
        final_pct = final_correct / total * 100 if total else 0
        delta = final_pct - init_pct

        logger.info(f"  {dataset_name.upper()} Results:")
        logger.info(f"    Initial EX: {initial_correct}/{total} = {init_pct:.2f}%")
        logger.info(f"    Final EX:   {final_correct}/{total} = {final_pct:.2f}%")
        logger.info(f"    Δ = {delta:+.2f}%")

        all_summaries[dataset_name] = {
            "total": total,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "initial_ex": init_pct,
            "final_ex": final_pct,
            "improvement": delta,
        }

        # Save per-dataset results
        out_path = output_dir / f"step4_{dataset_name}_results.json"
        with open(out_path, 'w') as f:
            json.dump({
                "dataset": dataset_name,
                "summary": all_summaries[dataset_name],
                "results": results,
            }, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "=" * 60)
    print("  Step 4: Pipeline Summary")
    print("=" * 60)
    for name, s in all_summaries.items():
        print(f"  {name.upper():8s}: {s['initial_ex']:.2f}% → {s['final_ex']:.2f}% (Δ={s['improvement']:+.2f}%)")
    print("=" * 60)

    # Save combined summary
    summary_path = output_dir / "step4_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"  Summary saved to {summary_path}")
    logger.info("✓ Step 4 完成")


if __name__ == "__main__":
    main()
