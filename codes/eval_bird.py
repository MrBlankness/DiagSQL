"""
DiagSQL - Evaluation Script (BIRD / Spider)

Evaluates SQL execution accuracy (EX) on BIRD or Spider dev set.
Can evaluate both raw predictions and pipeline results.

Usage:
  # Evaluate pipeline results (auto-detect dataset from results)
  python eval_bird.py --predictions ./pipeline_results/pipeline_results.json

  # Evaluate from a predict_dev.json file on BIRD
  python eval_bird.py --dataset bird --predict_json ./pipeline_results/predict_dev.json

  # Evaluate from a predict_dev.json file on Spider
  python eval_bird.py --dataset spider --predict_json ./pipeline_results/predict_dev.json
"""
import argparse
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ═══ Paths ═══
BIRD_DEV_JSON = "DiagSQL/dataset/BIRD/dev_20240627/dev.json"
BIRD_DB_PATH = "DiagSQL/dataset/BIRD/dev_20240627/dev_databases"
SPIDER_DEV_JSON = "DiagSQL/dataset/spider_data/dev.json"
SPIDER_DB_PATH = "DiagSQL/dataset/spider_data/database"

SQL_TIMEOUT = 30  # seconds


def execute_sql_for_acc(predicted_sql: str, ground_truth_sql: str, db_path: str) -> int:
    """Execute SQL and compare results. Returns 1 if match, 0 otherwise."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()

        cursor.execute(predicted_sql)
        predicted_res = set(map(tuple, cursor.fetchall()))

        cursor.execute(ground_truth_sql)
        ground_truth_res = set(map(tuple, cursor.fetchall()))

        conn.close()
        return 1 if predicted_res == ground_truth_res else 0
    except Exception:
        return 0


def evaluate_case(args_tuple) -> Dict:
    """Evaluate a single case (for multiprocessing)."""
    case_idx, predicted_sql, ground_truth_sql, db_path, timeout = args_tuple
    try:
        # Use a simple timeout mechanism
        import signal

        def handler(signum, frame):
            raise TimeoutError()

        # For systems that support signal.alarm (Unix)
        try:
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            result = execute_sql_for_acc(predicted_sql, ground_truth_sql, db_path)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            # Windows or threads: just run without timeout
            result = execute_sql_for_acc(predicted_sql, ground_truth_sql, db_path)

        return {"case_idx": case_idx, "acc": result}
    except Exception:
        return {"case_idx": case_idx, "acc": 0}


def evaluate_from_predictions(
    predictions: Dict[str, str],
    dev_data: List[Dict],
    db_base: str,
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: Dict mapping question_id (str) -> predicted SQL
        dev_data: Unified dev data (list of dicts with question_id, db_id, SQL, difficulty)
        db_base: Path to database directory

    Returns:
        Evaluation results dict
    """
    # Build task list
    tasks = []
    for item in dev_data:
        qid = str(item["question_id"])
        if qid not in predictions:
            continue

        pred_sql = predictions[qid]
        gt_sql = item["SQL"]
        db_id = item["db_id"]
        db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")
        difficulty = item.get("difficulty", "unknown")

        tasks.append({
            "question_id": qid,
            "predicted_sql": pred_sql,
            "ground_truth_sql": gt_sql,
            "db_path": db_path,
            "difficulty": difficulty,
        })

    # Execute evaluation
    results = []
    correct = 0
    difficulty_stats = {}

    for task in tqdm(tasks, desc="Evaluating"):
        if not os.path.exists(task["db_path"]):
            acc = 0
        else:
            acc = execute_sql_for_acc(
                task["predicted_sql"],
                task["ground_truth_sql"],
                task["db_path"],
            )

        results.append({
            "question_id": task["question_id"],
            "acc": acc,
            "difficulty": task["difficulty"],
        })
        correct += acc

        diff = task["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "correct": 0}
        difficulty_stats[diff]["total"] += 1
        difficulty_stats[diff]["correct"] += acc

    total = len(tasks)
    overall_acc = correct / total if total else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": overall_acc,
        "difficulty_stats": difficulty_stats,
        "per_case": results,
    }


def load_dev_data(dataset: str, dev_json: str) -> List[Dict]:
    """Load dev data and normalize to unified format."""
    with open(dev_json, 'r') as f:
        raw_data = json.load(f)

    unified = []
    for i, item in enumerate(raw_data):
        if dataset == "bird":
            unified.append({
                "question_id": item.get("question_id", i),
                "db_id": item["db_id"],
                "SQL": item["SQL"],
                "difficulty": item.get("difficulty", "unknown"),
            })
        elif dataset == "spider":
            unified.append({
                "question_id": i,
                "db_id": item["db_id"],
                "SQL": item["query"],  # Spider uses "query" field
                "difficulty": item.get("difficulty", "unknown"),
            })
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    return unified


def main(args):
    # Resolve dataset paths
    if args.dataset == "bird":
        dev_json = args.dev_json or BIRD_DEV_JSON
        db_path = args.db_path or BIRD_DB_PATH
    elif args.dataset == "spider":
        dev_json = args.dev_json or SPIDER_DEV_JSON
        db_path = args.db_path or SPIDER_DB_PATH
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load dev data
    print(f"Loading {args.dataset.upper()} dev data from {dev_json}")
    dev_data = load_dev_data(args.dataset, dev_json)
    print(f"Total dev samples: {len(dev_data)}")

    # Load predictions
    predictions = {}

    if args.predict_json:
        # Load from predict_dev.json format: {question_id: sql}
        print(f"Loading predictions from {args.predict_json}")
        with open(args.predict_json, 'r') as f:
            predictions = json.load(f)

    elif args.predictions:
        # Load from pipeline results format
        print(f"Loading pipeline results from {args.predictions}")
        with open(args.predictions, 'r') as f:
            data = json.load(f)

        # Auto-detect dataset from pipeline results if not explicitly set
        if data.get("summary", {}).get("dataset") and args.dataset == "bird":
            detected = data["summary"]["dataset"]
            if detected != args.dataset:
                print(f"  Auto-detected dataset from results: {detected}")
                args.dataset = detected
                if detected == "spider":
                    dev_json = args.dev_json or SPIDER_DEV_JSON
                    db_path = args.db_path or SPIDER_DB_PATH
                    dev_data = load_dev_data(args.dataset, dev_json)
                    print(f"  Reloaded {args.dataset.upper()} dev data: {len(dev_data)} samples")

        eval_field = args.eval_field  # "final_sql" or "initial_sql"
        for r in data.get("results", []):
            qid = str(r["question_id"])
            predictions[qid] = r.get(eval_field, r.get("final_sql", ""))

    else:
        print("Error: Must provide either --predict_json or --predictions")
        return

    print(f"Loaded {len(predictions)} predictions")

    # Evaluate
    eval_results = evaluate_from_predictions(predictions, dev_data, db_path)

    # Print results
    print("\n" + "=" * 70)
    print(f"  {args.dataset.upper()} Dev Set Evaluation Results")
    print("=" * 70)
    print(f"  Total evaluated:    {eval_results['total']}")
    print(f"  Correct:            {eval_results['correct']}")
    print(f"  Execution Accuracy: {eval_results['accuracy']*100:.2f}%")
    print()
    print(f"  By Difficulty:")
    for diff in sorted(eval_results["difficulty_stats"].keys()):
        s = eval_results["difficulty_stats"][diff]
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"    {diff:12s}: {s['correct']:4d}/{s['total']:4d} = {pct:.2f}%")
    print("=" * 70)

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIRD/Spider Evaluation")
    parser.add_argument("--dataset", type=str, default="bird", choices=["bird", "spider"],
                        help="Dataset: 'bird' or 'spider'")
    parser.add_argument("--predict_json", type=str, default=None,
                        help="Path to predict_dev.json (format: {qid: sql})")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to pipeline_results.json")
    parser.add_argument("--eval_field", type=str, default="final_sql",
                        choices=["initial_sql", "final_sql"],
                        help="Which SQL field to evaluate from pipeline results")
    parser.add_argument("--dev_json", type=str, default=None,
                        help="Path to dev JSON (auto-resolved from --dataset if not set)")
    parser.add_argument("--db_path", type=str, default=None,
                        help="Path to database directory (auto-resolved from --dataset if not set)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save detailed evaluation results")
    args = parser.parse_args()
    main(args)
