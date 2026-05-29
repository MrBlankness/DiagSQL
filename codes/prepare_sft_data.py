"""
Prepare SFT dataset from GPT-4o analysis results.

Sources:
  - gpt-4o-bird-log-8-final_results_gpt4o_analysis.json  (BIRD, per candidate)
  - sl_deepseek-r1_final_results_gpt4o_analysis.json      (Spider2, per candidate)
  - nl2sql_bugs_gpt4o_analysis.json                        (NL2SQL-Bugs, per entry)

Each sample:
  Input  → apply_chat_template([system, user])
  Output → JSON string of GPT-4o analysis result

User message uses SYSTEM_MESSAGE + USER_TEMPLATE from sql_selection_eval.py.
Filters out entries where analysis is None (GPT-4o call failed).

Output:
  sft_data/sft_train.jsonl   – raw {"messages": [...]} records (full conversations)
  sft_data/sft_stats.json    – dataset statistics
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("")
SFT_DATA_DIR = BASE / "sft" / "sft_data"

SOURCE_FILES = [
    BASE / "generate_sft_datasets" / "gpt-4o-bird-log-8-final_results_gpt4o_analysis.json",
    BASE / "generate_sft_datasets" / "sl_deepseek-r1_final_results_gpt4o_analysis.json",
    BASE / "generate_sft_datasets" / "nl2sql_bugs_gpt4o_analysis.json",
]

# ── Prompt templates (from sql_selection_eval.py) ──────────────────────────
SYSTEM_MESSAGE = (
    "You are a SQL error analysis expert. Given a user question, evidence, database schema, generated SQL query, and execution result, "
    "you need to analyze whether the SQL correctly answers the question and identify any error types if it's wrong.\n\n"
    "### Error Types Description:\n"
    "1. Attribute-Related Errors:\n"
    "- Attribute Mismatch: Wrong attribute selection\n"
    "- Attribute Redundancy: Unnecessary attributes included\n"
    "- Attribute Missing: Required attributes not included\n\n"
    "2. Table-Related Errors:\n"
    "- Table Mismatch: Wrong table selection\n"
    "- Table Redundancy: Unnecessary tables included\n"
    "- Table Missing: Required tables not included\n"
    "- Join Condition Mismatch: The join condition between tables is incorrect\n"
    "- Join Type Mismatch: Wrong join type (e.g., LEFT JOIN when INNER JOIN needed)\n\n"
    "3. Value-Related Errors:\n"
    "- Value Mismatch: Incorrect values in conditions\n"
    "- Data Format Mismatch: Wrong data type or format\n\n"
    "4. Operator-Related Errors:\n"
    "- Comparison Operator Mismatch: Wrong operator (e.g., = when != needed)\n"
    "- Logical Operator Mismatch: Wrong logical operator (e.g., AND when OR needed)\n\n"
    "5. Condition-Related Errors:\n"
    "- Explicit Condition Missing: Required conditions not included\n"
    "- Explicit Condition Mismatch: Wrong conditions\n"
    "- Explicit Condition Redundancy: Unnecessary conditions\n"
    "- Implicit Condition Missing: Missing implicit conditions\n\n"
    "6. Function-Related Errors:\n"
    "- Aggregate Functions: Incorrect use of SUM, AVG, etc.\n"
    "- Window Functions: Incorrect use of OVER, PARTITION BY, etc.\n"
    "- Date/Time Functions: Incorrect use of JULIANDAY, strftime, etc.\n"
    "- Conversion Functions: Incorrect use of CAST, etc.\n"
    "- Math Functions: Incorrect use of ROUND, etc.\n"
    "- String Functions: Incorrect use of SUBSTR, etc.\n"
    "- Conditional Functions: Incorrect use of IIF, CASE WHEN, etc.\n\n"
    "7. Clause-Related Errors:\n"
    "- Clause Missing: Required clauses not included\n"
    "- Clause Redundancy: Unnecessary clauses included\n\n"
    "8. Subquery-Related Errors:\n"
    "- Subquery Missing: Required subquery not included\n"
    "- Subquery Mismatch: Incorrect subquery logic\n"
    "- Partial Query: The error SQL query is a partial query\n\n"
    "9. Other Errors:\n"
    "- ASC/DESC: Incorrect sorting order\n"
    "- DISTINCT: Incorrect use of DISTINCT\n"
    "- Other: Major logical errors requiring complete rewrite\n\n"
    "### Output Format:\n"
    "You must output your analysis in the following JSON format:\n"
    "- If the SQL is correct, output: {\"label\": \"right\"}\n"
    "- If the SQL is wrong, output: {\n"
    "        \"label\": \"wrong\",\n"
    "        \"error_types\": [\n"
    "            {\n"
    "                \"reasoning\": \"explanation of why this error type applies\",\n"
    "                \"error_type\": \"error category from the above list\",\n"
    "                \"sub_error_type\": \"specific error type from the above list\"\n"
    "            }\n"
    "        ]\n"
    "    }\n"
)

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


def build_user_message(question: str, evidence: str, schema: str,
                        sql: str, execution_result: str) -> str:
    return USER_TEMPLATE.format(
        question=question or "",
        evidence=evidence or "N/A",
        schema=schema or "N/A",
        sql=sql or "",
        execution_result=execution_result or "N/A",
    )


def analysis_to_output(analysis) -> Optional[str]:
    """Convert analysis dict (or bare list) to a clean JSON string for the target output."""
    if analysis is None:
        return None

    # Some entries have analysis as a bare list of error_types (GPT-4o formatting quirk)
    if isinstance(analysis, list):
        error_types_raw = analysis
        label = "wrong"
    elif isinstance(analysis, dict):
        label = analysis.get("label")
        if label not in ("right", "wrong"):
            return None
        if label == "right":
            return json.dumps({"label": "right"}, ensure_ascii=False)
        error_types_raw = analysis.get("error_types", [])
    else:
        return None

    # wrong: must have at least one error_type with reasoning
    if not error_types_raw:
        return None
    # Keep only entries with all required fields and non-empty reasoning
    clean_ets = []
    for et in error_types_raw:
        if not isinstance(et, dict):
            continue
        reasoning = (et.get("reasoning") or "").strip()
        error_type = (et.get("error_type") or "").strip()
        sub_error_type = (et.get("sub_error_type") or "").strip()
        if reasoning and error_type and sub_error_type:
            clean_ets.append({
                "reasoning": reasoning,
                "error_type": error_type,
                "sub_error_type": sub_error_type,
            })
    if not clean_ets:
        return None
    return json.dumps({"label": "wrong", "error_types": clean_ets}, ensure_ascii=False)


def process_bird_or_spider2(path: Path) -> List[Dict]:
    """
    Process gpt-4o-bird / sl_deepseek-r1 analysis JSON.
    Structure: results[].candidates[].{question, evidence, schema, generated_sql,
               generated_execution_result, analysis}
    """
    logger.info(f"Processing {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    skipped_no_analysis = 0
    skipped_bad_output = 0

    for result in data["results"]:
        candidates = result.get("candidates", [])
        for cand in candidates:
            analysis = cand.get("analysis")
            if analysis is None:
                skipped_no_analysis += 1
                continue

            output_str = analysis_to_output(analysis)
            if output_str is None:
                skipped_bad_output += 1
                continue

            question = cand.get("question", "")
            evidence = cand.get("evidence", "")
            schema = cand.get("schema", "")
            sql = cand.get("generated_sql", "")
            exec_result = cand.get("generated_execution_result", "")

            if not question or not sql:
                skipped_bad_output += 1
                continue

            inferred_label = "wrong" if isinstance(analysis, list) else analysis.get("label", "wrong")
            user_msg = build_user_message(question, evidence, schema, sql, exec_result)
            samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_MESSAGE},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": output_str},
                ],
                "source": path.stem,
                "label": inferred_label,
            })

    logger.info(
        f"  {path.name}: {len(samples)} samples kept, "
        f"{skipped_no_analysis} skipped (no analysis), "
        f"{skipped_bad_output} skipped (bad output)"
    )
    return samples


def process_nl2sql_bugs(path: Path) -> List[Dict]:
    """
    Process nl2sql_bugs analysis JSON.
    Structure: results[].{question, evidence, schema, generated_sql,
               generated_execution_result, analysis}
    """
    logger.info(f"Processing {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    skipped_no_analysis = 0
    skipped_bad_output = 0

    for entry in data["results"]:
        analysis = entry.get("analysis")
        if analysis is None:
            skipped_no_analysis += 1
            continue

        output_str = analysis_to_output(analysis)
        if output_str is None:
            skipped_bad_output += 1
            continue

        question = entry.get("question", "")
        evidence = entry.get("evidence", "")
        schema = entry.get("schema", "")
        sql = entry.get("generated_sql", "")
        exec_result = entry.get("generated_execution_result", "")

        if not question or not sql:
            skipped_bad_output += 1
            continue

        inferred_label = "wrong" if isinstance(analysis, list) else analysis.get("label", "wrong")
        user_msg = build_user_message(question, evidence, schema, sql, exec_result)
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_MESSAGE},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": output_str},
            ],
            "source": path.stem,
            "label": inferred_label,
        })

    logger.info(
        f"  {path.name}: {len(samples)} samples kept, "
        f"{skipped_no_analysis} skipped (no analysis), "
        f"{skipped_bad_output} skipped (bad output)"
    )
    return samples


def main():
    os.makedirs(SFT_DATA_DIR, exist_ok=True)
    random.seed(42)

    all_samples = []

    for path in SOURCE_FILES:
        if not path.exists():
            logger.error(f"Source file not found: {path}")
            continue
        if "nl2sql_bugs" in path.stem:
            samples = process_nl2sql_bugs(path)
        else:
            samples = process_bird_or_spider2(path)
        all_samples.extend(samples)

    # ── Filter by token length (≤ MAX_SEQ_LENGTH) ─────────────────────────
    MAX_SEQ_LENGTH = 4096
    QWEN3_MODEL = "Qwen3-8B"
    logger.info(f"Filtering samples by token length (max={MAX_SEQ_LENGTH}) using Qwen3 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    filtered = []
    too_long = 0
    for s in all_samples:
        try:
            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False
            )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens <= MAX_SEQ_LENGTH:
            filtered.append(s)
        else:
            too_long += 1
    logger.info(f"  Kept {len(filtered)}/{len(all_samples)} samples, dropped {too_long} (too long)")
    all_samples = filtered

    # ── Shuffle ──────────────────────────────────────────────────────────
    random.shuffle(all_samples)
    logger.info(f"Total samples after shuffle: {len(all_samples)}")

    # ── Split train / val (95% / 5%) ─────────────────────────────────────
    n_val = max(100, int(len(all_samples) * 0.05))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    # ── Label distribution ────────────────────────────────────────────────
    def count_labels(samples):
        right = sum(1 for s in samples if s["label"] == "right")
        wrong = sum(1 for s in samples if s["label"] == "wrong")
        return right, wrong

    train_r, train_w = count_labels(train_samples)
    val_r, val_w = count_labels(val_samples)

    # ── Save ──────────────────────────────────────────────────────────────
    train_path = SFT_DATA_DIR / "sft_train.jsonl"
    val_path   = SFT_DATA_DIR / "sft_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_samples),
        "max_seq_length_filter": MAX_SEQ_LENGTH,
        "dropped_too_long": too_long,
        "train": {"total": len(train_samples), "right": train_r, "wrong": train_w},
        "val":   {"total": len(val_samples),   "right": val_r,   "wrong": val_w},
        "sources": {},
    }
    for path in SOURCE_FILES:
        src_name = path.stem
        stats["sources"][src_name] = sum(1 for s in all_samples if s["source"] == src_name)

    stats_path = SFT_DATA_DIR / "sft_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Train: {len(train_samples)} samples  (right={train_r}, wrong={train_w})")
    logger.info(f"Val:   {len(val_samples)} samples  (right={val_r}, wrong={val_w})")
    logger.info(f"Saved train → {train_path}")
    logger.info(f"Saved val   → {val_path}")
    logger.info(f"Saved stats → {stats_path}")


if __name__ == "__main__":
    main()
