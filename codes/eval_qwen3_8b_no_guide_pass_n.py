"""
Evaluate Qwen3-8B model on NL2SQL-Bugs-with-evidence.json with pass@n metric.
Based on eval_qwen3_8b_with_guide_pass_n.py, but WITHOUT per-case identification
guidance (no judge_guide.json is used).

pass@n definition:
    Each case is run n times (default n=8) with temperature sampling.
    A case is considered "correct" if AT LEAST ONE of the n responses
    has the correct label.

Acceleration via vLLM:
    - SamplingParams(n=N): prefill is computed ONCE per prompt; N completions
      are sampled from the same KV-cache → huge speedup vs repeating prompts N times.
    - PagedAttention + continuous batching handle the rest automatically.
    - llm.chat() accepts raw message dicts, applies Qwen3 chat-template internally.
    - No manual mini-batch loop required; vLLM schedules everything internally.

Additional metrics reported:
    - pass@n label accuracy / binary metrics (primary)
    - mean@n  label accuracy (average single-run accuracy across all n runs)
    - best-of-n error-type F1 (oracle: for gt-wrong cases, pick the n
      response with highest error-type F1 among those predicting "wrong")

Usage:
    python eval_qwen3_8b_no_guide_pass_n.py \
        [--n_samples 8] [--temperature 0.7] [--top_p 0.9] \
        [--tensor_parallel_size 1] [--gpu_memory_utilization 0.90] \
        [--max_samples N] [--output_path out.json]
"""

import json
import re
import sqlite3
import logging
import argparse
import threading
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────── constants ────────────────────────────────

MODEL_PATH = ""
DATA_PATH    = "DiagSQL/dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence.json"
DB_BASE_PATH = "DiagSQL/dataset/BIRD/dev_20240627/dev_databases"

DEFAULT_N              = 8      # pass@n  —  number of completions per prompt

# ── Local DB cache directory ───────────────────────────────────────────────
# We copy every SQLite file from the CephFS remote mount into a local /tmp
# directory at startup.  All subsequent schema reads and SQL executions use
# the local copy, so the remote mount is touched exactly once per database
# (the initial cp) rather than thousands of times during the prompt-build loop.
# This is the single most effective fix for the "Transport endpoint is not
# connected" errors that previously caused the CephFS mount to drop.
_LOCAL_DB_DIR: Optional[Path] = None   # set by _init_local_db_cache()


def _init_local_db_cache(db_base: str, db_ids: List[str]) -> Path:
    """
    Copy the required SQLite files from *db_base* (CephFS) into a local
    temp directory.  Returns the local directory path.

    Called ONCE at startup, before any schema reads or SQL executions.
    """
    global _LOCAL_DB_DIR
    local_dir = Path(tempfile.mkdtemp(prefix="nl2sql_dbs_"))
    logger.info(f"Copying {len(db_ids)} SQLite DBs to local cache: {local_dir}")
    for db_id in db_ids:
        src = Path(db_base) / db_id / f"{db_id}.sqlite"
        dst_dir = local_dir / db_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{db_id}.sqlite"
        if src.exists():
            shutil.copy2(str(src), str(dst))
            logger.info(f"  Copied: {db_id} ({src.stat().st_size // 1024 // 1024} MB)")
        else:
            logger.warning(f"  NOT FOUND on remote: {src}")
    _LOCAL_DB_DIR = local_dir
    logger.info(f"Local DB cache ready: {local_dir}")
    return local_dir
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.9

# ───────────────────────────── prompt templates ───────────────────────────

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

# ───────────────────────────── DB utilities ───────────────────────────────

# ──────────────────────── schema cache (per db_id) ────────────────────────
# Fix: avoid reading the same SQLite schema 2018 times over CephFS.
# Each db_id is read ONCE and cached in memory.
_schema_cache: Dict[str, str] = {}
_schema_cache_lock = threading.Lock()


def _resolve_db_path(db_id: str, db_base: str) -> Path:
    """
    Return the path to the SQLite file for *db_id*.
    If the local DB cache has been initialised (_LOCAL_DB_DIR is set) and
    the local copy exists, return that path instead of the CephFS path.
    This is the key fix: all DB access after _init_local_db_cache() goes
    through local /tmp storage, so the CephFS mount is never hit again.
    """
    if _LOCAL_DB_DIR is not None:
        local = _LOCAL_DB_DIR / db_id / f"{db_id}.sqlite"
        if local.exists():
            return local
    return Path(db_base) / db_id / f"{db_id}.sqlite"


def get_database_schema(db_id: str, db_base: str = DB_BASE_PATH) -> str:
    """Return CREATE TABLE DDL string for all tables in the SQLite DB.

    Results are cached in memory (keyed by db_id) so each database is opened
    at most ONCE.  After _init_local_db_cache() the file is read from the
    local /tmp copy, not from CephFS.
    """
    if not db_id:
        return ""

    with _schema_cache_lock:
        if db_id in _schema_cache:
            return _schema_cache[db_id]

    sqlite_path = _resolve_db_path(db_id, db_base)
    if not sqlite_path.exists():
        logger.warning(f"SQLite file not found: {sqlite_path}")
        with _schema_cache_lock:
            _schema_cache[db_id] = ""
        return ""

    schema_info = ""
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                columns = cursor.fetchall()
                cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
                foreign_keys = cursor.fetchall()

                col_defs     = []
                primary_keys = []
                for col in columns:
                    col_id, col_name, col_type, not_null, default_val, pk = col
                    col_def = f"    {col_name} {col_type or 'TEXT'}"
                    if not_null:
                        col_def += " NOT NULL"
                    if default_val is not None:
                        col_def += f" DEFAULT {default_val}"
                    col_defs.append(col_def)
                    if pk > 0:
                        primary_keys.append(col_name)

                if primary_keys:
                    col_defs.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")
                for fk in foreign_keys:
                    _, _, to_table, from_col, to_col, _, _, _ = fk
                    col_defs.append(
                        f"    FOREIGN KEY (`{table_name}`.`{from_col}`) REFERENCES `{to_table}`.`{to_col}`"
                    )

                schema_info += f"CREATE TABLE `{table_name}` (\n"
                schema_info += ",\n".join(col_defs)
                schema_info += "\n);\n\n"
            schema_info = schema_info.strip()
        finally:
            conn.close()   # always release the file handle
    except Exception as e:
        logger.error(f"Schema extraction error for {db_id}: {e}")
        schema_info = ""

    with _schema_cache_lock:
        _schema_cache[db_id] = schema_info
    return schema_info


# ──────────────────── SQL execution with timeout guard ────────────────────────
# Runaway SQL queries can hold a file handle open indefinitely.
# We abort queries that take longer than SQL_EXEC_TIMEOUT_S seconds.
SQL_EXEC_TIMEOUT_S = 5   # seconds; increase if your queries legitimately run long


def _interrupted_handler(conn):
    """Called from a timer thread to interrupt a long-running SQLite query."""
    try:
        conn.interrupt()
    except Exception:
        pass


def execute_sql_query(
    db_id: str, sql: str, limit: int = 100, db_base: str = DB_BASE_PATH
) -> Dict[str, Any]:
    """Execute SQL on the SQLite DB and return result dict.

    Uses _resolve_db_path() so after startup all queries go to the local /tmp
    copy of the database, completely avoiding CephFS I/O during the prompt-
    build loop.  A threading.Timer watchdog aborts runaway queries after
    SQL_EXEC_TIMEOUT_S seconds and the connection is always closed in finally.
    """
    if not db_id or not sql:
        return {"success": False, "error": "Missing db_id or sql"}
    sqlite_path = _resolve_db_path(db_id, db_base)
    if not sqlite_path.exists():
        return {"success": False, "error": f"SQLite file not found: {sqlite_path}"}
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10, check_same_thread=False)
        cursor = conn.cursor()
        sql_exec = sql.rstrip(";")
        if "limit" not in sql_exec.lower():
            sql_exec = f"{sql_exec} LIMIT {limit}"
        # Watchdog: interrupt long-running queries
        timer = threading.Timer(SQL_EXEC_TIMEOUT_S, _interrupted_handler, args=(conn,))
        timer.start()
        cursor.execute(sql_exec)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return {"success": True, "columns": cols, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def format_execution_result(result: Dict[str, Any]) -> str:
    if result["success"]:
        s = f"Success: {result['row_count']} rows returned"
        if result["rows"]:
            s += f"\nSample data: {result['rows'][:3]}"
        return s
    return f"Error: {result['error']}"

# ───────────────────────────── prompt builder ─────────────────────────────

def build_prompt_text(
    item: Dict[str, Any],
    tokenizer,
    db_base: str = DB_BASE_PATH,
) -> str:
    """
    Build the fully-rendered prompt string for one data item.
    Uses the HuggingFace tokenizer's apply_chat_template with
    enable_thinking=False to prepend <think>\\n\\n</think>\\n\\n,
    which suppresses Qwen3's chain-of-thought output.
    The resulting string is passed directly to llm.generate().
    No per-case identification guidance is included.
    """
    schema      = get_database_schema(item["db_id"], db_base)
    exec_result = execute_sql_query(item["db_id"], item["sql"], db_base=db_base)
    exec_str    = format_execution_result(exec_result)

    user_msg = USER_TEMPLATE.format(
        question=item["question"],
        evidence=item.get("evidence", ""),
        schema=schema,
        sql=item["sql"],
        execution_result=exec_str,
    )
    msgs = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": user_msg},
    ]
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3: prepend <think>\n\n</think>\n\n
            )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e}, using fallback")
    # fallback: plain text concatenation
    return f"System: {SYSTEM_MESSAGE}\n\nUser: {user_msg}\n\nAssistant:"

# ─────────────────────────── response parsing ─────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    code_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_fence:
        try:
            return json.loads(code_fence.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_response(response_text: str) -> Tuple[Optional[str], List[Dict]]:
    """
    Returns (pred_label, pred_error_types).
    pred_label: 'right' / 'wrong' / None (parse failure).
    """
    parsed = extract_json(response_text)
    if parsed is None:
        return None, []
    label = parsed.get("label")
    if label not in ("right", "wrong"):
        return None, []
    error_types = parsed.get("error_types", [])
    if not isinstance(error_types, list):
        error_types = []
    return label, error_types

# ──────────────────────── pass@n evaluation helpers ───────────────────────

def error_type_f1(
    pred_error_types: List[Dict],
    gold_error_types: List[Dict],
) -> Tuple[float, float, float]:
    """Compute (precision, recall, f1) for a single prediction vs gold."""
    pred_set = {
        (e.get("error_type", ""), e.get("sub_error_type", ""))
        for e in pred_error_types
    }
    gold_set = {
        (e.get("error_type", ""), e.get("sub_error_type", ""))
        for e in gold_error_types
    }
    tp_set    = pred_set & gold_set
    precision = len(tp_set) / len(pred_set) if pred_set else 0.0
    recall    = len(tp_set) / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return precision, recall, f1


def select_best_wrong_prediction(
    run_labels: List[Optional[str]],
    run_error_types: List[List[Dict]],
    gold_error_types: List[Dict],
) -> Tuple[Optional[List[Dict]], float, float, float]:
    """
    Among all runs that predicted 'wrong', pick the one with the highest
    error-type F1 against gold_error_types.
    Returns (best_pred_error_types, best_precision, best_recall, best_f1).
    Returns (None, 0, 0, 0) if no run predicted 'wrong'.
    """
    best_f1   = -1.0
    best_pred = None
    best_p    = 0.0
    best_r    = 0.0
    for label, ets in zip(run_labels, run_error_types):
        if label == "wrong":
            p, r, f1 = error_type_f1(ets, gold_error_types)
            if f1 > best_f1:
                best_f1   = f1
                best_pred = ets
                best_p    = p
                best_r    = r
    if best_pred is None:
        return None, 0.0, 0.0, 0.0
    return best_pred, best_p, best_r, best_f1

# ─────────────────────────────── metrics ──────────────────────────────────

@dataclass
class PassAtNMetrics:
    """Accumulates metrics for pass@n and mean@n evaluation."""
    n: int = DEFAULT_N

    # ── sample counts ──
    total: int = 0
    right_total: int = 0    # gt = right
    wrong_total: int = 0    # gt = wrong

    # ── pass@n label ──
    pass_n_correct:   int = 0   # pass@n: ≥1 run correct
    pass_n_right_tp:  int = 0   # gt=right AND pass@n correct
    pass_n_wrong_tp:  int = 0   # gt=wrong AND pass@n correct

    # ── mean@n label (average single-run accuracy) ──
    mean_n_correct_sum:   float = 0.0   # sum of per-sample correct-run fraction
    mean_n_right_tp_sum:  float = 0.0
    mean_n_wrong_tp_sum:  float = 0.0

    # ── parse stats ──
    total_runs:       int = 0   # n * total
    parse_fail_runs:  int = 0

    # ── error-type: best-of-n (oracle) ──
    bon_et_precision_sum: float = 0.0
    bon_et_recall_sum:    float = 0.0
    bon_et_f1_sum:        float = 0.0
    bon_et_count:         int   = 0     # gt-wrong cases with ≥1 "wrong" pred

    bon_et_stats: Dict = field(
        default_factory=lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )

    # ── error-type: mean-of-n per-sub_error_type TP/FP/FN ──
    mon_et_stats: Dict = field(
        default_factory=lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )

    # ── error-type: mean-of-n (average per-run ET F1 for gt-wrong ∩ pred-wrong) ──
    mon_et_precision_sum: float = 0.0
    mon_et_recall_sum:    float = 0.0
    mon_et_f1_sum:        float = 0.0
    mon_et_count:         int   = 0     # individual runs that are gt-wrong ∩ pred-wrong

    def summary(self) -> Dict[str, Any]:
        total = self.total or 1

        # ── pass@n label metrics ──
        pass_acc = self.pass_n_correct / total

        tp_p = self.pass_n_wrong_tp
        fp_p = self.right_total - self.pass_n_right_tp
        fn_p = self.wrong_total - self.pass_n_wrong_tp
        tn_p = self.pass_n_right_tp

        pass_bin_p  = tp_p / (tp_p + fp_p) if (tp_p + fp_p) else 0.0
        pass_bin_r  = tp_p / (tp_p + fn_p) if (tp_p + fn_p) else 0.0
        pass_bin_f1 = (
            2 * pass_bin_p * pass_bin_r / (pass_bin_p + pass_bin_r)
            if (pass_bin_p + pass_bin_r) else 0.0
        )
        pass_spec   = tn_p / (tn_p + fp_p) if (tn_p + fp_p) else 0.0

        # ── mean@n label metrics ──
        mean_acc = self.mean_n_correct_sum / total

        # mean@n TP/FP/FN/TN (fractional)
        tp_m = self.mean_n_wrong_tp_sum
        fp_m = self.right_total - self.mean_n_right_tp_sum
        fn_m = self.wrong_total - self.mean_n_wrong_tp_sum
        tn_m = self.mean_n_right_tp_sum

        mean_bin_p  = tp_m / (tp_m + fp_m) if (tp_m + fp_m) else 0.0
        mean_bin_r  = tp_m / (tp_m + fn_m) if (tp_m + fn_m) else 0.0
        mean_bin_f1 = (
            2 * mean_bin_p * mean_bin_r / (mean_bin_p + mean_bin_r)
            if (mean_bin_p + mean_bin_r) else 0.0
        )
        mean_spec   = tn_m / (tn_m + fp_m) if (tn_m + fp_m) else 0.0

        # ── error-type: best-of-n ──
        bon_p  = self.bon_et_precision_sum / self.bon_et_count if self.bon_et_count else 0.0
        bon_r  = self.bon_et_recall_sum    / self.bon_et_count if self.bon_et_count else 0.0
        bon_f1 = self.bon_et_f1_sum        / self.bon_et_count if self.bon_et_count else 0.0

        # ── error-type: mean-of-n ──
        mon_p  = self.mon_et_precision_sum / self.mon_et_count if self.mon_et_count else 0.0
        mon_r  = self.mon_et_recall_sum    / self.mon_et_count if self.mon_et_count else 0.0
        mon_f1 = self.mon_et_f1_sum        / self.mon_et_count if self.mon_et_count else 0.0

        parse_fail_rate = self.parse_fail_runs / self.total_runs if self.total_runs else 0.0

        return {
            # ── overview ──
            "n_samples_per_case":             self.n,
            "total_cases":                    self.total,
            "total_runs":                     self.total_runs,
            "parse_fail_runs":                self.parse_fail_runs,
            "parse_fail_rate":                round(parse_fail_rate, 4),
            "right_cases (gt)":               self.right_total,
            "wrong_cases (gt)":               self.wrong_total,
            # ── pass@n ──
            f"pass@{self.n}_label_accuracy":              round(pass_acc,    4),
            f"pass@{self.n}_binary_precision":            round(pass_bin_p,  4),
            f"pass@{self.n}_binary_recall (sensitivity)": round(pass_bin_r,  4),
            f"pass@{self.n}_binary_f1":                   round(pass_bin_f1, 4),
            f"pass@{self.n}_specificity":                 round(pass_spec,   4),
            f"pass@{self.n}_confusion_matrix":
                f"TP={tp_p} FP={fp_p} FN={fn_p} TN={tn_p}",
            # ── mean@n ──
            f"mean@{self.n}_label_accuracy":              round(mean_acc,    4),
            f"mean@{self.n}_binary_precision":            round(mean_bin_p,  4),
            f"mean@{self.n}_binary_recall (sensitivity)": round(mean_bin_r,  4),
            f"mean@{self.n}_binary_f1":                   round(mean_bin_f1, 4),
            f"mean@{self.n}_specificity":                 round(mean_spec,   4),
            # ── error-type best-of-n (oracle) ──
            "best_of_n_et_avg_precision":     round(bon_p,  4),
            "best_of_n_et_avg_recall":        round(bon_r,  4),
            "best_of_n_et_avg_f1":            round(bon_f1, 4),
            "best_of_n_et_eval_count":        self.bon_et_count,
            # ── error-type mean-of-n ──
            "mean_of_n_et_avg_precision":     round(mon_p,  4),
            "mean_of_n_et_avg_recall":        round(mon_r,  4),
            "mean_of_n_et_avg_f1":            round(mon_f1, 4),
            "mean_of_n_et_eval_count":        self.mon_et_count,
        }


def update_pass_at_n_metrics(
    metrics: PassAtNMetrics,
    true_label: bool,
    run_labels: List[Optional[str]],          # length n, each 'right'/'wrong'/None
    run_error_types: List[List[Dict]],        # length n
    gold_error_types: List[Dict],
) -> None:
    """Update PassAtNMetrics for one case with its n run results."""
    metrics.total      += 1
    metrics.total_runs += len(run_labels)

    true_str = "right" if true_label else "wrong"

    if true_label:
        metrics.right_total += 1
    else:
        metrics.wrong_total += 1

    # ── parse failures ──
    n_parse_fail = sum(1 for l in run_labels if l is None)
    metrics.parse_fail_runs += n_parse_fail

    valid_runs = [(l, et) for l, et in zip(run_labels, run_error_types) if l is not None]

    # ── pass@n: correct if ≥1 valid run matches true label ──
    pass_correct = any(l == true_str for l, _ in valid_runs)
    if pass_correct:
        metrics.pass_n_correct += 1
        if true_label:
            metrics.pass_n_right_tp += 1
        else:
            metrics.pass_n_wrong_tp += 1

    # ── mean@n: fraction of valid runs that are correct ──
    if valid_runs:
        frac_correct = sum(1 for l, _ in valid_runs if l == true_str) / len(run_labels)
        metrics.mean_n_correct_sum += frac_correct
        if true_label:
            metrics.mean_n_right_tp_sum += sum(
                1 for l, _ in valid_runs if l == "right"
            ) / len(run_labels)
        else:
            metrics.mean_n_wrong_tp_sum += sum(
                1 for l, _ in valid_runs if l == "wrong"
            ) / len(run_labels)

    # ── error-type metrics (only for gt-wrong cases) ──
    if not true_label and gold_error_types:

        # best-of-n (oracle): pick highest-F1 response among those predicting "wrong"
        best_pred_ets, bon_p, bon_r, bon_f1 = select_best_wrong_prediction(
            run_labels, run_error_types, gold_error_types
        )
        if best_pred_ets is not None:
            metrics.bon_et_precision_sum += bon_p
            metrics.bon_et_recall_sum    += bon_r
            metrics.bon_et_f1_sum        += bon_f1
            metrics.bon_et_count         += 1

            # accumulate per-sub_error_type stats using best prediction
            gold_set = {
                (e.get("error_type", ""), e.get("sub_error_type", ""))
                for e in gold_error_types
            }
            pred_set = {
                (e.get("error_type", ""), e.get("sub_error_type", ""))
                for e in best_pred_ets
            }
            for et in (pred_set & gold_set):
                metrics.bon_et_stats[et[1]]["tp"] += 1
            for et in (pred_set - gold_set):
                metrics.bon_et_stats[et[1]]["fp"] += 1
            for et in (gold_set - pred_set):
                metrics.bon_et_stats[et[1]]["fn"] += 1

        # mean-of-n: average over individual runs that predicted "wrong"
        for label, pred_ets in zip(run_labels, run_error_types):
            if label == "wrong":
                p, r, f1 = error_type_f1(pred_ets, gold_error_types)
                metrics.mon_et_precision_sum += p
                metrics.mon_et_recall_sum    += r
                metrics.mon_et_f1_sum        += f1
                metrics.mon_et_count         += 1

                # accumulate per-sub_error_type stats for mean-of-n
                gold_set = {
                    (e.get("error_type", ""), e.get("sub_error_type", ""))
                    for e in gold_error_types
                }
                pred_set = {
                    (e.get("error_type", ""), e.get("sub_error_type", ""))
                    for e in pred_ets
                }
                for et in (pred_set & gold_set):
                    metrics.mon_et_stats[et[1]]["tp"] += 1
                for et in (pred_set - gold_set):
                    metrics.mon_et_stats[et[1]]["fp"] += 1
                for et in (gold_set - pred_set):
                    metrics.mon_et_stats[et[1]]["fn"] += 1

# ──────────────────────────── vLLM inference ──────────────────────────────

def vllm_generate(
    llm: LLM,
    all_prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[List[str]]:
    """
    Run vLLM inference with SamplingParams(n=n).

    Key acceleration points:
      - Each prompt is prefilled ONCE; n completions are sampled from the
        same KV-cache → ~n× cheaper prefill vs expanding prompts n times.
      - PagedAttention manages memory efficiently for large batches.
      - Continuous batching maximises GPU utilisation across all prompts.
      - Prompts are pre-rendered strings (with enable_thinking=False already
        applied), passed to llm.generate() for maximum compatibility.

    Args:
        llm:           Initialised vLLM LLM instance.
        all_prompts:   List of pre-rendered prompt strings, one per case.
        n:             Number of completions to sample per prompt.
        max_new_tokens: Max output tokens per completion.
        temperature:   Sampling temperature (use >0 for diversity).
        top_p:         Nucleus sampling threshold.

    Returns:
        List[List[str]]: outer list = cases, inner list = n completions.
    """
    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        skip_special_tokens=True,
    )

    logger.info(
        f"Submitting {len(all_prompts)} prompts to vLLM  "
        f"(n={n}, temperature={temperature}, top_p={top_p}, "
        f"max_tokens={max_new_tokens}) ..."
    )

    # llm.generate() accepts pre-rendered strings directly
    outputs = llm.generate(
        prompts=all_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    # outputs[i].outputs is a list of n CompletionOutput objects
    results: List[List[str]] = []
    for req_output in outputs:
        completions = [comp.text.strip() for comp in req_output.outputs]
        results.append(completions)

    return results

# ─────────────────────────────── main ────────────────────────────────────

def main(args):
    n = args.n_samples
    logger.info(f"pass@{n} evaluation  (temperature={args.temperature}, top_p={args.top_p})")

    # ── 1. Load data ──
    logger.info(f"Loading dataset from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples and args.max_samples < len(data):
        data = data[: args.max_samples]
        logger.info(f"Using first {args.max_samples} samples")

    logger.info(
        f"Dataset size: {len(data)} cases  "
        f"(right={sum(1 for d in data if d['label'])}, "
        f"wrong={sum(1 for d in data if not d['label'])})"
    )

    # ── 2. Load tokenizer (for apply_chat_template with enable_thinking=False) ──
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    logger.info("Tokenizer loaded.")

    # ── 3. Build prompt strings (schema + SQL execution, no guide) ──
    # CRITICAL FIX: copy all SQLite databases from CephFS to local /tmp FIRST.
    # After this step every get_database_schema() and execute_sql_query() call
    # goes to /tmp — CephFS is never touched again during the prompt-build loop.
    unique_dbs = sorted({item["db_id"] for item in data})
    _init_local_db_cache(args.db_base_path, unique_dbs)

    logger.info("Pre-warming schema cache for all unique db_ids (local copies) ...")
    for db_id in unique_dbs:
        _ = get_database_schema(db_id, args.db_base_path)
        logger.info(f"  Schema cached: {db_id}")
    logger.info(f"Schema cache warmed for {len(unique_dbs)} databases.")

    logger.info("Building prompt strings (schema + SQL execution, no guide) ...")
    all_prompts: List[str] = []
    for i, item in enumerate(data):
        prompt = build_prompt_text(item, tokenizer, args.db_base_path)
        all_prompts.append(prompt)
        if (i + 1) % 200 == 0:
            logger.info(f"  Built {i+1}/{len(data)} prompts")
    logger.info(f"All {len(all_prompts)} prompts built.  CephFS I/O complete.")

    # ── 4. Initialise vLLM engine ──
    logger.info(
        f"Initialising vLLM engine  "
        f"(model={args.model_path}, "
        f"tensor_parallel_size={args.tensor_parallel_size}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}) ..."
    )
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Allow long contexts (schema + SQL can be large)
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        # vLLM will automatically detect and use the model's chat template
        # This is handled via pre-rendered prompts with enable_thinking=False
    )
    logger.info("vLLM engine ready.")

    # ── 5. Inference ──
    # SamplingParams(n=n): each prompt is prefilled ONCE, then n completions
    # are sampled from the same KV-cache — far more efficient than repeating
    # the same prompt n times.
    logger.info(
        f"Running vLLM inference  "
        f"(n={n}, max_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, top_p={args.top_p}) ..."
    )
    case_responses: List[List[str]] = vllm_generate(
        llm=llm,
        all_prompts=all_prompts,
        n=n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    logger.info(
        f"Inference complete.  {len(case_responses)} cases × {n} completions "
        f"= {len(case_responses) * n} total responses."
    )
    assert len(case_responses) == len(data), (
        f"Expected {len(data)} case results, got {len(case_responses)}"
    )

    # ── 6. Evaluate ──
    logger.info("Evaluating responses ...")
    metrics = PassAtNMetrics(n=n)
    results = []

    for i, item in enumerate(data):
        true_label       = item["label"]           # bool
        gold_error_types = item.get("error_types", [])

        run_responses = case_responses[i]          # list of n completion strings
        run_labels:      List[Optional[str]] = []
        run_error_types: List[List[Dict]]    = []
        for resp in run_responses:
            lbl, ets = parse_response(resp)
            run_labels.append(lbl)
            run_error_types.append(ets)

        update_pass_at_n_metrics(
            metrics, true_label, run_labels, run_error_types, gold_error_types
        )

        true_str   = "right" if true_label else "wrong"
        pass_n_hit = any(l == true_str for l in run_labels if l is not None)

        if not true_label:
            best_pred_ets, _, _, _ = select_best_wrong_prediction(
                run_labels, run_error_types, gold_error_types
            )
        else:
            best_pred_ets = None

        results.append({
            "id":                    item.get("id", ""),
            "question":              item["question"],
            "db_id":                 item["db_id"],
            "sql":                   item["sql"],
            "true_label":            true_label,
            "true_error_types":      gold_error_types,
            f"pass@{n}_correct":     pass_n_hit,
            "runs": [
                {
                    "run_index":        j,
                    "pred_label":       run_labels[j],
                    "pred_error_types": run_error_types[j],
                    "raw_response":     run_responses[j],
                }
                for j in range(n)
            ],
            "best_of_n_pred_error_types": best_pred_ets,
        })

    # ── 7. Print summary ──
    summary = metrics.summary()
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY  —  Qwen3-8B  pass@{n}  on NL2SQL-Bugs  (no guide, vLLM)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:<55s}: {v}")

    if metrics.bon_et_stats:
        print("\n" + "=" * 70)
        print(f"PER-SUB_ERROR_TYPE STATS  (best-of-{n}, gt-wrong cases with ≥1 'wrong' pred)")
        print(f"  {'sub_error_type':<45s}  TP   FP   FN    P      R     F1")
        print("-" * 70)
        for sub_et, cnt in sorted(metrics.bon_et_stats.items()):
            tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            print(f"  {sub_et:<45s} {tp:3d}  {fp:3d}  {fn:3d}  {p:.3f}  {r:.3f}  {f1:.3f}")

    if metrics.mon_et_stats:
        print("\n" + "=" * 70)
        print(f"PER-SUB_ERROR_TYPE STATS  (mean-of-{n}, gt-wrong ∩ pred-wrong individual runs)")
        print(f"  {'sub_error_type':<45s}  TP   FP   FN    P      R     F1")
        print("-" * 70)
        for sub_et, cnt in sorted(metrics.mon_et_stats.items()):
            tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            print(f"  {sub_et:<45s} {tp:3d}  {fp:3d}  {fn:3d}  {p:.3f}  {r:.3f}  {f1:.3f}")

    # ── 8. Save results ──
    if args.output_path:
        def et_stats_with_prf(stats_dict):
            """Convert raw TP/FP/FN dict to include P, R, F1 for each sub_error_type."""
            out = {}
            for sub_et, cnt in stats_dict.items():
                tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
                p  = tp / (tp + fp) if (tp + fp) else 0.0
                r  = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                out[sub_et] = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 4),
                    "recall":    round(r, 4),
                    "f1":        round(f1, 4),
                }
            return out

        out = {
            "model":   args.model_path,
            "dataset": args.data_path,
            "args":    vars(args),
            "summary": summary,
            f"per_sub_error_type_stats_best_of_{n}": et_stats_with_prf(metrics.bon_et_stats),
            f"per_sub_error_type_stats_mean_of_{n}": et_stats_with_prf(metrics.mon_et_stats),
            "results": results,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved → {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-8B on NL2SQL-Bugs with pass@n metric, "
                    "without per-case identification guidance, "
                    "and vLLM-accelerated inference."
    )
    parser.add_argument("--model_path",              default=MODEL_PATH)
    parser.add_argument("--data_path",               default=DATA_PATH)
    parser.add_argument("--db_base_path",            default=DB_BASE_PATH)
    parser.add_argument("--output_path",             default="eval_results_qwen3_8b_no_guide_pass_n_v6_best.json",
                        help="Path to save detailed results JSON")
    # pass@n
    parser.add_argument("--n_samples",               type=int,   default=DEFAULT_N,
                        help=f"Number of completions per prompt for pass@n (default {DEFAULT_N})")
    parser.add_argument("--temperature",             type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top_p",                   type=float, default=DEFAULT_TOP_P,
                        help=f"Top-p nucleus sampling (default {DEFAULT_TOP_P})")
    parser.add_argument("--max_new_tokens",          type=int,   default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Max output tokens per completion (default {DEFAULT_MAX_NEW_TOKENS})")
    # vLLM engine
    parser.add_argument("--tensor_parallel_size",    type=int,   default=1,
                        help="Number of GPUs for tensor parallelism (default 1)")
    parser.add_argument("--gpu_memory_utilization",  type=float, default=0.90,
                        help="Fraction of GPU memory vLLM may use (default 0.90)")
    parser.add_argument("--max_model_len",           type=int,   default=8192,
                        help="Maximum sequence length (prompt + output) (default 8192)")
    # dataset
    parser.add_argument("--max_samples",             type=int,   default=None,
                        help="Limit dataset size for quick debug (e.g. --max_samples 20)")
    args = parser.parse_args()
    main(args)
