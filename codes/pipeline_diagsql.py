"""
DiagSQL Closed-Loop Pipeline: GPT-4o SQL Generation → DiagSQL Validation → Iterative Refinement

Pipeline:
  1. GPT-4o generates SQL from (question, evidence, schema)
  2. DiagSQL validator diagnoses the SQL (correct/wrong + error types)
  3. If SQL is wrong, feed diagnosis back to GPT-4o for refinement
  4. Repeat up to K rounds

Supports both BIRD and Spider datasets.

Usage:
  # BIRD dataset (default)
  python pipeline_diagsql.py --dataset bird \
      --diagsql_model_path <path_to_diagsql_checkpoint> \
      --max_refine_rounds 3 --max_samples 50

  # Spider dataset
  python pipeline_diagsql.py --dataset spider \
      --diagsql_model_path <path_to_diagsql_checkpoint> \
      --max_refine_rounds 3 --max_samples 50
"""
import argparse
import json
import os
import re
import sqlite3
import time
import hashlib
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
BIRD_DEV_JSON = "DiagSQL/dataset/BIRD/dev_20240627/dev.json"
BIRD_DB_PATH = "DiagSQL/dataset/BIRD/dev_20240627/dev_databases"
SPIDER_DEV_JSON = "DiagSQL/dataset/spider_data/dev.json"
SPIDER_DB_PATH = "DiagSQL/dataset/spider_data/database"
DEFAULT_DIAGSQL_MODEL = "checkpoints/qwen3-8b-sql-sft/final_merged"

# ═══════════════════════════════════════════════════════════════════════════
# GPT-4o API Client
# ═══════════════════════════════════════════════════════════════════════════

class GPT4oClient:
    """GPT-4o API client."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.token = ""
        self.url = ""

    def call(self, messages: List[Dict], max_retries: int = 3) -> str:
        """Call GPT-4o API with retry logic."""
        for attempt in range(max_retries):
            try:
                payload = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': self.temperature,
                }
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.token}',
                }
                response = requests.post(
                    self.url, headers=headers,
                    data=json.dumps(payload), timeout=120
                )
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                return content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    logger.warning(f"GPT-4o API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"GPT-4o API failed after {max_retries} attempts: {e}")
                    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Database Utilities
# ═══════════════════════════════════════════════════════════════════════════

_schema_cache: Dict[str, str] = {}


def get_database_schema(db_id: str, db_base: str = BIRD_DB_PATH) -> str:
    """Extract database schema DDL."""
    if db_id in _schema_cache:
        return _schema_cache[db_id]

    sqlite_path = Path(db_base) / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        return ""

    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()

        schema_parts = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            foreign_keys = cursor.fetchall()

            col_defs = []
            pk_cols = []
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                col_def = f"    {col_name} {col_type or 'TEXT'}"
                if not_null:
                    col_def += " NOT NULL"
                col_defs.append(col_def)
                if pk > 0:
                    pk_cols.append(col_name)

            if pk_cols:
                col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
            for fk in foreign_keys:
                _, _, to_table, from_col, to_col, *_ = fk
                col_defs.append(f"    FOREIGN KEY ({from_col}) REFERENCES {to_table}({to_col})")

            schema_parts.append(f"CREATE TABLE `{table_name}` (\n" + ",\n".join(col_defs) + "\n);")

        conn.close()
        result = "\n\n".join(schema_parts)
        _schema_cache[db_id] = result
        return result
    except Exception as e:
        logger.error(f"Schema error for {db_id}: {e}")
        return ""


def execute_sql(db_id: str, sql: str, db_base: str = BIRD_DB_PATH, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL and return result."""
    if not db_id or not sql:
        return {"success": False, "error": "Missing db_id or sql"}

    sqlite_path = Path(db_base) / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        return {"success": False, "error": f"DB not found: {sqlite_path}"}

    conn = None
    timer = None
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10)
        cursor = conn.cursor()
        sql_exec = sql.rstrip(";")
        if "limit" not in sql_exec.lower():
            sql_exec = f"{sql_exec} LIMIT 100"

        timer = threading.Timer(timeout, lambda c: c.interrupt(), args=(conn,))
        timer.start()
        cursor.execute(sql_exec)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return {"success": True, "columns": cols, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if timer:
            timer.cancel()
        if conn:
            try:
                conn.close()
            except:
                pass


def format_execution_result(result: Dict[str, Any]) -> str:
    """Format SQL execution result as string."""
    if result["success"]:
        s = f"Success: {result['row_count']} rows returned"
        if result["rows"]:
            sample = [dict(zip(result.get("columns", []), r)) for r in result["rows"][:3]]
            s += f"\nSample data: {sample}"
        return s
    return f"Error: {result['error']}"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: GPT-4o SQL Generation
# ═══════════════════════════════════════════════════════════════════════════

SQL_GEN_SYSTEM = """You are an expert SQL developer. Given a natural language question, evidence (hints), and a database schema, generate a correct SQLite SQL query that answers the question.

Important:
- Output ONLY the SQL query, no explanation.
- Use SQLite syntax.
- Do not include markdown code fences.
"""

SQL_GEN_USER_TEMPLATE = """Question: {question}
Evidence: {evidence}
Database Schema:
{schema}

Generate the SQL query that answers the question. Output ONLY the SQL query."""


def generate_sql(client: GPT4oClient, question: str, evidence: str, schema: str) -> str:
    """Use GPT-4o to generate SQL."""
    messages = [
        {"role": "system", "content": SQL_GEN_SYSTEM},
        {"role": "user", "content": SQL_GEN_USER_TEMPLATE.format(
            question=question, evidence=evidence, schema=schema
        )},
    ]
    response = client.call(messages)
    # Extract SQL from response (remove markdown fences if present)
    sql = response.strip()
    sql = re.sub(r'^```(?:sql)?\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    return sql.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: DiagSQL Validation (using vLLM)
# ═══════════════════════════════════════════════════════════════════════════

DIAGSQL_SYSTEM = (
    "You are a SQL error analysis expert. Given a user question, evidence, database schema, generated SQL query, and execution result, "
    "you need to analyze whether the SQL correctly answers the question and identify any error types if it's wrong.\n\n"
    "### Error Types Description:\n"
    "1. Attribute-Related Errors:\n- Attribute Mismatch\n- Attribute Redundancy\n- Attribute Missing\n\n"
    "2. Table-Related Errors:\n- Table Mismatch\n- Table Redundancy\n- Table Missing\n- Join Condition Mismatch\n- Join Type Mismatch\n\n"
    "3. Value-Related Errors:\n- Value Mismatch\n- Data Format Mismatch\n\n"
    "4. Operator-Related Errors:\n- Comparison Operator Mismatch\n- Logical Operator Mismatch\n\n"
    "5. Condition-Related Errors:\n- Explicit Condition Missing\n- Explicit Condition Mismatch\n- Explicit Condition Redundancy\n- Implicit Condition Missing\n\n"
    "6. Function-Related Errors:\n- Aggregate Functions\n- Window Functions\n- Date/Time Functions\n- Conversion Functions\n- Math Functions\n- String Functions\n- Conditional Functions\n\n"
    "7. Clause-Related Errors:\n- Clause Missing\n- Clause Redundancy\n\n"
    "8. Subquery-Related Errors:\n- Subquery Missing\n- Subquery Mismatch\n- Partial Query\n\n"
    "9. Other Errors:\n- ASC/DESC\n- DISTINCT\n- Other\n\n"
    "### Output Format:\n"
    'You must output your analysis in the following JSON format:\n'
    '- If the SQL is correct, output: {"label": "right"}\n'
    '- If the SQL is wrong, output: {"label": "wrong", "error_types": [{"reasoning": "...", "error_type": "...", "sub_error_type": "..."}]}\n'
)

DIAGSQL_USER_TEMPLATE = (
    "Question: {question}\n"
    "Evidence: {evidence}\n"
    "Database Schema:\n{schema}\n"
    "Generated SQL: {sql}\n"
    "Execution Result: {execution_result}\n\n"
    "Please analyze whether this SQL query correctly answers the question. "
    "If incorrect, identify the specific error types."
)


class DiagSQLValidator:
    """DiagSQL validator using vLLM for inference."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 4,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 8192):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        logger.info(f"Loading DiagSQL model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.1,
            top_p=0.95,
            skip_special_tokens=True,
        )
        logger.info("DiagSQL validator loaded.")

    def diagnose(self, question: str, evidence: str, schema: str,
                 sql: str, execution_result: str) -> Dict[str, Any]:
        """Diagnose a single SQL query."""
        user_msg = DIAGSQL_USER_TEMPLATE.format(
            question=question, evidence=evidence, schema=schema,
            sql=sql, execution_result=execution_result,
        )
        messages = [
            {"role": "system", "content": DIAGSQL_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except:
            prompt = f"System: {DIAGSQL_SYSTEM}\n\nUser: {user_msg}\n\nAssistant:"

        outputs = self.llm.generate([prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        return self._parse_diagnosis(response_text)

    def diagnose_batch(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Batch diagnose multiple SQL queries."""
        prompts = []
        for item in items:
            user_msg = DIAGSQL_USER_TEMPLATE.format(
                question=item["question"],
                evidence=item["evidence"],
                schema=item["schema"],
                sql=item["sql"],
                execution_result=item["execution_result"],
            )
            messages = [
                {"role": "system", "content": DIAGSQL_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except:
                prompt = f"System: {DIAGSQL_SYSTEM}\n\nUser: {user_msg}\n\nAssistant:"
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(self._parse_diagnosis(text))
        return results

    def _parse_diagnosis(self, text: str) -> Dict[str, Any]:
        """Parse DiagSQL response into structured result."""
        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "label" in parsed:
                return parsed
        except:
            pass

        # Try to extract JSON from text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "label" in parsed:
                    return parsed
            except:
                pass

        # Fallback: try to find label
        if '"label"' in text:
            if '"right"' in text:
                return {"label": "right"}
            elif '"wrong"' in text:
                return {"label": "wrong", "error_types": [], "raw": text}

        return {"label": "unknown", "error_types": [], "raw": text}


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: GPT-4o Refinement with Diagnosis Feedback
# ═══════════════════════════════════════════════════════════════════════════

REFINE_SYSTEM = """You are an expert SQL developer. You previously generated an SQL query that was diagnosed as incorrect. Based on the diagnosis, refine your SQL query to fix the identified errors.

Important:
- Output ONLY the corrected SQL query, no explanation.
- Use SQLite syntax.
- Do not include markdown code fences.
"""

REFINE_USER_TEMPLATE = """Question: {question}
Evidence: {evidence}
Database Schema:
{schema}

Previous SQL (diagnosed as incorrect):
{previous_sql}

Diagnosis:
{diagnosis}

Based on the diagnosis above, generate a corrected SQL query. Output ONLY the SQL query."""


def refine_sql(client: GPT4oClient, question: str, evidence: str, schema: str,
               previous_sql: str, diagnosis: Dict) -> str:
    """Use GPT-4o to refine SQL based on DiagSQL diagnosis."""
    # Format diagnosis for the prompt
    diag_str = f"Label: {diagnosis.get('label', 'wrong')}\n"
    error_types = diagnosis.get("error_types", [])
    if error_types:
        diag_str += "Identified errors:\n"
        for i, et in enumerate(error_types, 1):
            reasoning = et.get("reasoning", "")
            error_type = et.get("error_type", "")
            sub_error_type = et.get("sub_error_type", "")
            diag_str += f"  {i}. [{error_type} - {sub_error_type}]: {reasoning}\n"

    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": REFINE_USER_TEMPLATE.format(
            question=question, evidence=evidence, schema=schema,
            previous_sql=previous_sql, diagnosis=diag_str,
        )},
    ]
    response = client.call(messages)
    sql = response.strip()
    sql = re.sub(r'^```(?:sql)?\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    return sql.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation: Execution Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_execution_accuracy(predicted_sql: str, ground_truth_sql: str,
                                db_id: str, db_base: str = BIRD_DB_PATH,
                                timeout: int = 30) -> int:
    """Compare predicted SQL vs ground truth by execution results. Returns 1 if match, 0 otherwise."""
    sqlite_path = Path(db_base) / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        return 0

    conn = None
    timer = None
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10)
        cursor = conn.cursor()

        timer = threading.Timer(timeout, lambda c: c.interrupt(), args=(conn,))
        timer.start()

        cursor.execute(predicted_sql)
        predicted_res = set(map(tuple, cursor.fetchall()))

        cursor.execute(ground_truth_sql)
        ground_truth_res = set(map(tuple, cursor.fetchall()))

        return 1 if predicted_res == ground_truth_res else 0
    except Exception:
        return 0
    finally:
        if timer:
            timer.cancel()
        if conn:
            try:
                conn.close()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Loading (BIRD / Spider)
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name: str, dev_json: str, db_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load dataset in a unified format.

    Returns list of dicts with keys:
        question_id, question, evidence, db_id, ground_truth_sql, difficulty
    """
    with open(dev_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if max_samples and max_samples < len(raw_data):
        raw_data = raw_data[:max_samples]

    unified = []
    for i, item in enumerate(raw_data):
        if dataset_name == "bird":
            unified.append({
                "question_id": item.get("question_id", i),
                "question": item["question"],
                "evidence": item.get("evidence", ""),
                "db_id": item["db_id"],
                "ground_truth_sql": item["SQL"],
                "difficulty": item.get("difficulty", "unknown"),
            })
        elif dataset_name == "spider":
            unified.append({
                "question_id": i,
                "question": item["question"],
                "evidence": "",  # Spider has no evidence
                "db_id": item["db_id"],
                "ground_truth_sql": item["query"],  # Spider uses "query" field
                "difficulty": item.get("difficulty", "unknown"),  # Spider may not have difficulty
            })
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'bird' or 'spider'.")

    logger.info(f"Loaded {len(unified)} samples from {dataset_name} dataset ({dev_json})")
    return unified


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Run the full DiagSQL closed-loop pipeline."""

    # Resolve dataset paths
    if args.dataset == "bird":
        dev_json = args.dev_json or BIRD_DEV_JSON
        db_path = args.db_path or BIRD_DB_PATH
    elif args.dataset == "spider":
        dev_json = args.dev_json or SPIDER_DEV_JSON
        db_path = args.db_path or SPIDER_DB_PATH
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load dataset
    data = load_dataset(args.dataset, dev_json, db_path, args.max_samples)
    logger.info(f"Processing {len(data)} samples, dataset={args.dataset}, max_refine_rounds={args.max_refine_rounds}")

    # Initialize GPT-4o client
    gpt_client = GPT4oClient(model=args.gpt_model, temperature=args.gpt_temperature)

    # Initialize DiagSQL validator
    validator = DiagSQLValidator(
        model_path=args.diagsql_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Phase 1: Generate initial SQL with GPT-4o ═══
    logger.info("Phase 1: Generating SQL with GPT-4o...")
    results = []
    for i, item in enumerate(tqdm(data, desc="SQL Generation")):
        question = item["question"]
        evidence = item["evidence"]
        db_id = item["db_id"]

        schema = get_database_schema(db_id, db_path)

        # Generate SQL
        generated_sql = generate_sql(gpt_client, question, evidence, schema)

        results.append({
            "question_id": item["question_id"],
            "question": question,
            "evidence": evidence,
            "db_id": db_id,
            "difficulty": item["difficulty"],
            "ground_truth_sql": item["ground_truth_sql"],
            "schema": schema,
            "initial_sql": generated_sql,
            "current_sql": generated_sql,
            "refinement_history": [],
            "final_sql": generated_sql,
        })

    # ═══ Phase 2: Iterative Diagnosis + Refinement ═══
    for round_idx in range(args.max_refine_rounds):
        logger.info(f"\nPhase 2 - Refinement Round {round_idx + 1}/{args.max_refine_rounds}")

        # Collect items that need diagnosis (not yet confirmed as "right")
        items_to_diagnose = []
        for idx, r in enumerate(results):
            if r.get("diagnosed_right", False):
                continue
            items_to_diagnose.append((idx, r))

        if not items_to_diagnose:
            logger.info("All SQL queries diagnosed as correct. Stopping early.")
            break

        logger.info(f"  Diagnosing {len(items_to_diagnose)} SQL queries...")

        # Batch prepare diagnosis inputs
        diag_inputs = []
        for idx, r in items_to_diagnose:
            exec_result = execute_sql(r["db_id"], r["current_sql"], db_path)
            exec_str = format_execution_result(exec_result)
            diag_inputs.append({
                "question": r["question"],
                "evidence": r["evidence"],
                "schema": r["schema"],
                "sql": r["current_sql"],
                "execution_result": exec_str,
            })

        # Batch diagnosis
        diagnoses = validator.diagnose_batch(diag_inputs)

        # Process diagnoses and refine where needed
        refine_count = 0
        for (idx, r), diagnosis in zip(items_to_diagnose, diagnoses):
            r["refinement_history"].append({
                "round": round_idx + 1,
                "sql": r["current_sql"],
                "diagnosis": diagnosis,
            })

            if diagnosis.get("label") == "right":
                r["diagnosed_right"] = True
                continue

            # SQL diagnosed as wrong → refine
            refine_count += 1
            new_sql = refine_sql(
                gpt_client,
                question=r["question"],
                evidence=r["evidence"],
                schema=r["schema"],
                previous_sql=r["current_sql"],
                diagnosis=diagnosis,
            )
            r["current_sql"] = new_sql
            r["final_sql"] = new_sql

        logger.info(f"  Round {round_idx + 1}: {refine_count} queries refined, "
                    f"{len(items_to_diagnose) - refine_count} confirmed correct")

    # ═══ Phase 3: Evaluate ═══
    logger.info("\nPhase 3: Evaluating execution accuracy...")
    correct_count = 0
    difficulty_stats = {}

    for r in tqdm(results, desc="Evaluation"):
        # Evaluate initial SQL
        initial_acc = evaluate_execution_accuracy(
            r["initial_sql"], r["ground_truth_sql"], r["db_id"], db_path
        )
        # Evaluate final SQL (after refinement)
        final_acc = evaluate_execution_accuracy(
            r["final_sql"], r["ground_truth_sql"], r["db_id"], db_path
        )

        r["initial_correct"] = initial_acc
        r["final_correct"] = final_acc
        correct_count += final_acc

        # Track by difficulty
        diff = r["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "initial_correct": 0, "final_correct": 0}
        difficulty_stats[diff]["total"] += 1
        difficulty_stats[diff]["initial_correct"] += initial_acc
        difficulty_stats[diff]["final_correct"] += final_acc

    # ═══ Results Summary ═══
    total = len(results)
    initial_correct = sum(r["initial_correct"] for r in results)
    final_correct = correct_count

    print("\n" + "=" * 70)
    print(f"  DiagSQL Closed-Loop Pipeline Results ({args.dataset.upper()})")
    print("=" * 70)
    print(f"  Dataset:                 {args.dataset}")
    print(f"  Total samples:           {total}")
    print(f"  Max refine rounds (K):   {args.max_refine_rounds}")
    print(f"  GPT-4o model:            {args.gpt_model}")
    print(f"  DiagSQL model:           {args.diagsql_model_path}")
    print(f"")
    print(f"  Initial EX (GPT-4o):     {initial_correct}/{total} = {initial_correct/total*100:.2f}%")
    print(f"  Final EX (after refine): {final_correct}/{total} = {final_correct/total*100:.2f}%")
    print(f"  Improvement:             +{(final_correct-initial_correct)/total*100:.2f}%")
    print(f"")
    print(f"  By Difficulty:")
    for diff in sorted(difficulty_stats.keys()):
        s = difficulty_stats[diff]
        init_pct = s["initial_correct"] / s["total"] * 100 if s["total"] else 0
        final_pct = s["final_correct"] / s["total"] * 100 if s["total"] else 0
        print(f"    {diff:12s}: {s['final_correct']}/{s['total']} = {final_pct:.2f}% "
              f"(initial: {init_pct:.2f}%, Δ={final_pct-init_pct:+.2f}%)")
    print("=" * 70)

    # Save results
    output_file = output_dir / "pipeline_results.json"
    summary = {
        "config": vars(args),
        "dataset": args.dataset,
        "total_samples": total,
        "initial_ex": initial_correct / total if total else 0,
        "final_ex": final_correct / total if total else 0,
        "improvement": (final_correct - initial_correct) / total if total else 0,
        "difficulty_stats": difficulty_stats,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Save SQL predictions
    pred_file = output_dir / "predict_dev.json"
    predictions = {}
    for r in results:
        predictions[str(r["question_id"])] = r["final_sql"]
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiagSQL Closed-Loop Pipeline (BIRD / Spider)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="bird", choices=["bird", "spider"],
                        help="Dataset to evaluate on: 'bird' or 'spider'")

    # Model paths
    parser.add_argument("--diagsql_model_path", type=str, default=DEFAULT_DIAGSQL_MODEL,
                        help="Path to DiagSQL trained model checkpoint")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o",
                        help="GPT model for SQL generation and refinement")
    parser.add_argument("--gpt_temperature", type=float, default=0.3)

    # Pipeline parameters
    parser.add_argument("--max_refine_rounds", type=int, default=3,
                        help="Maximum number of refinement rounds (K)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debug)")

    # Data paths (override defaults based on --dataset)
    parser.add_argument("--dev_json", type=str, default=None,
                        help="Path to dev JSON file (auto-resolved from --dataset if not set)")
    parser.add_argument("--db_path", type=str, default=None,
                        help="Path to database directory (auto-resolved from --dataset if not set)")

    # vLLM settings for DiagSQL
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)

    # Output
    parser.add_argument("--output_dir", type=str, default="./pipeline_results")

    args = parser.parse_args()
    run_pipeline(args)
