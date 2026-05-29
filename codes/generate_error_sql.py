#!/usr/bin/env python3
"""
Full production version for generating NL2SQL error data based on LLM.

This script reads correct NL2SQL data, calls LLM to generate erroneous SQL
statements, and annotates error types for training error detection models.

Acceleration strategy:
  - asyncio + openai.AsyncOpenAI for concurrent LLM requests
  - Semaphore limits concurrency to MAX_CONCURRENCY (matches vLLM max-num-seqs)
  - Schema pre-loading and caching to avoid repeated DB I/O
  - Demo examples pre-loaded once at startup
  - max_tokens reduced to a reasonable value for SQL generation
"""

import json
import random
import sqlite3
import asyncio
import re
import ijson
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

random.seed(2026)

start_index = 0
end_index = 2000

# ---------- Path configuration ----------
DATA_FILE = "../datasets/OmniSQL/data.json"
DEMO_FILE = "../dataset/NL2SQLBugs/NL2SQL-Bugs-grouped-by-error-types.json"
BIRD_FILE = "../datasets/BIRD/dev_20240627/dev.json"
BIRD_DATABASE_DIR = "../datasets/BIRD/dev_20240627/dev_databases"
SynSQL_DATABASE_DIR = "../datasets/OmniSQL/databases"
OUTPUT_FILE = f"../dataset/NL2SQLBugs/generated_error_data_2k.json"

# ---------- Concurrency configuration ----------
# Match vLLM --max-num-seqs 16; tune down if OOM occurs
MAX_CONCURRENCY = 16

# ---------- Error type description (shared across prompts) ----------
ERROR_TYPES_DESCRIPTION = """
### Error Types Description:
1. Attribute-Related Errors:
- Attribute Mismatch: Wrong attribute selection
- Attribute Redundancy: Unnecessary attributes included
- Attribute Missing: Required attributes not included

2. Table-Related Errors:
- Table Mismatch: Wrong table selection
- Table Redundancy: Unnecessary tables included
- Table Missing: Required tables not included
- Join Condition Mismatch: The join condition between tables is incorrect
- Join Type Mismatch: Wrong join type (e.g., LEFT JOIN when INNER JOIN needed)

3. Value-Related Errors:
- Value Mismatch: Incorrect values in conditions
- Data Format Mismatch: Wrong data type or format

4. Operator-Related Errors:
- Comparison Operator Mismatch: Wrong operator (e.g., = when != needed)
- Logical Operator Mismatch: Wrong logical operator (e.g., AND when OR needed)

5. Condition-Related Errors:
- Explicit Condition Missing: Required conditions not included
- Explicit Condition Mismatch: Wrong conditions
- Explicit Condition Redundancy: Unnecessary conditions
- Implicit Condition Missing: Missing implicit conditions

6. Function-Related Errors:
- Aggregate Functions: Incorrect use of SUM, AVG, etc.
- Window Functions: Incorrect use of OVER, PARTITION BY, etc.
- Date/Time Functions: Incorrect use of date/time functions
- Conversion Functions: Incorrect use of CAST, etc.
- Math Functions: Incorrect use of ROUND, etc.
- String Functions: Incorrect use of SUBSTR, etc.
- Conditional Functions: Incorrect use of IIF, CASE WHEN, etc.

7. Clause-Related Errors:
- Clause Missing: Required clauses not included
- Clause Redundancy: Unnecessary clauses included

8. Subquery-Related Errors:
- Subquery Missing: Required subquery not included
- Subquery Mismatch: Incorrect subquery logic
- Partial Query: The error SQL query is a partial query

9. Other Errors:
- ASC/DESC: Incorrect sorting order
- DISTINCT: Incorrect use of DISTINCT
- Other: Major logical errors requiring complete rewrite
"""


# ─────────────────────────────────────────────
# Schema helpers  (with in-memory cache)
# ─────────────────────────────────────────────
_schema_cache: dict[str, str] = {}


def get_database_schema(data_type: str, db_id: str) -> str:
    """Return DDL-style schema string for a SQLite database (cached)."""
    cache_key = f"{data_type}::{db_id}"
    if cache_key in _schema_cache:
        return _schema_cache[cache_key]

    if data_type == "SynSQL":
        db_path = Path(SynSQL_DATABASE_DIR) / db_id / f"{db_id}.sqlite"
    elif data_type == "BIRD":
        db_path = Path(BIRD_DATABASE_DIR) / db_id / f"{db_id}.sqlite"
    else:
        return f"Unknown data_type: {data_type}"

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_info = f"Database Schema for {db_id}:\n\n"

        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            schema_info += f"CREATE TABLE {table_name} (\n"
            for col in columns:
                col_name, col_type, not_null, default_val, pk = (
                    col[1], col[2], col[3], col[4], col[5],
                )
                schema_info += f"    {col_name} {col_type}"
                if not_null:
                    schema_info += " NOT NULL"
                if pk:
                    schema_info += " PRIMARY KEY"
                if default_val:
                    schema_info += f" DEFAULT {default_val}"
                schema_info += ",\n"

            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            for fk in foreign_keys:
                schema_info += f"    FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]}),\n"

            schema_info = schema_info.rstrip(",\n") + "\n);\n\n"

        conn.close()
        _schema_cache[cache_key] = schema_info
        return schema_info

    except Exception as e:
        msg = f"Error reading schema for {db_id}: {str(e)}"
        _schema_cache[cache_key] = msg
        return msg


# ─────────────────────────────────────────────
# Demo example helpers  (pre-loaded once)
# ─────────────────────────────────────────────
_demo_pool: list[dict] = []


def preload_demo_examples() -> None:
    """Pre-load all demo examples (with schema) into memory once."""
    global _demo_pool
    print("Pre-loading demo examples …")
    try:
        with open(DEMO_FILE, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
        with open(BIRD_FILE, "r", encoding="utf-8") as f:
            bird_data = json.load(f)

        # Build a fast lookup: question -> correct SQL
        bird_lookup: dict[str, str] = {
            item["question"]: item["SQL"] for item in bird_data
        }

        for group in demo_data:
            for case in group.get("cases", []):
                example = dict(case)
                example["error_sql"] = example["sql"]
                example["sql"] = bird_lookup.get(example.get("question", ""), "")
                example["schema_info"] = get_database_schema(
                    "BIRD", example.get("db_id", "")
                )
                _demo_pool.append(example)

        print(f"Loaded {len(_demo_pool)} demo examples.")
    except Exception as e:
        print(f"Error pre-loading demo examples: {e}")


def get_demo_example() -> dict | None:
    """Return a random demo example from the pre-loaded pool."""
    if not _demo_pool:
        return None
    return random.choice(_demo_pool)


# ─────────────────────────────────────────────
# Prompt builder  (English)
# ─────────────────────────────────────────────
def generate_error_sql_prompt(
    correct_data: dict, demo_example: dict, schema_info: str
) -> str:
    """Build an English prompt asking the LLM to produce one erroneous SQL."""

    num_errors = random.randint(1, 5)

    prompt = f"""You are an expert SQL analyst. Your task is to introduce realistic errors into a correct SQL query and annotate the error types precisely.

## Task Description
Given a correct SQL query, generate **1 erroneous SQL statement** that contains **{num_errors} error(s)**.
The errors should look plausible — avoid obvious syntax mistakes; prefer semantic or logical errors.

## Error Type Reference
{ERROR_TYPES_DESCRIPTION}

## Demonstration Example
**Question**: {demo_example.get('question', '')}
**External Knowledge**: {demo_example.get('evidence', '')}
**Database**: {demo_example.get('db_id', '')}
**Database Schema**:
{demo_example.get('schema_info', '')}
**Correct SQL**: {demo_example.get('sql', '')}
**Erroneous SQL**: {demo_example.get('error_sql', '')}
**Error Types**: {json.dumps(demo_example.get('error_types', []), indent=2, ensure_ascii=False)}

## Current Task
**Question**: {correct_data.get('question', '')}
**External Knowledge**: {correct_data.get('external_knowledge', '')}
**Database ID**: {correct_data.get('db_id', '')}
**Database Schema**:
{schema_info}
**Correct SQL**: {correct_data.get('sql', '')}

## Requirements
1. Generate exactly **1** erroneous SQL statement containing **{num_errors}** error(s).
2. Errors must be plausible and non-trivial (avoid pure syntax errors).
3. Annotate each error with the exact category and sub-category from the reference above.
4. Use sub-category level granularity for error types.

## Output Format
Respond with **only** a JSON object (no markdown fences) in the following structure:
{{
  "error_sql": {{
    "sql": "<the erroneous SQL statement>",
    "error_types": [
      {{"error_type": "<Category>", "sub_error_type": "<Specific Error>"}},
      ...
    ]
  }},
  "explanation": "<brief description of what errors were introduced and why>"
}}
"""
    return prompt


# ─────────────────────────────────────────────
# Async LLM call
# ─────────────────────────────────────────────
_async_client: AsyncOpenAI | None = None


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    return _async_client


async def call_llm_async(prompt: str, semaphore: asyncio.Semaphore) -> str | None:
    """Call the LLM asynchronously, respecting the concurrency semaphore.

    Notes
    -----
    * ``enable_thinking=False`` disables Qwen3's chain-of-thought reasoning tokens
      so that the entire ``max_tokens`` budget is used for the actual JSON response.
      Without this, reasoning tokens can consume thousands of tokens and leave the
      model no room to finish the JSON.
    * ``max_tokens=4096`` gives ample room for the JSON output while staying well
      within the 65 k context limit.
    """
    async with semaphore:
        client = get_async_client()
        try:
            response = await client.chat.completions.create(
                model="Qwen3-32B",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                extra_body={"enable_thinking": False},  # disable CoT reasoning tokens
            )
            content = response.choices[0].message.content
            if content is None:
                print("LLM returned None content (likely all reasoning tokens)")
                return None
            return content.strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None


# ─────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────
def parse_llm_response(llm_response: str) -> dict | None:
    """Parse the LLM JSON response, with a regex fallback.

    Returns None if the response is empty, un-parseable, or contains an
    empty / missing SQL string (quality filter).
    """
    if not llm_response:
        return None

    data = None
    try:
        data = json.loads(llm_response)
    except json.JSONDecodeError:
        # Try to extract the outermost JSON object
        json_match = re.search(
            r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", llm_response
        )
        if json_match:
            try:
                data = json.loads(json_match.group())
            except Exception:
                pass
        if data is None:
            print(f"Failed to parse LLM response as JSON:\n{llm_response[:300]}")
            return None

    # Quality filter: discard results where the error SQL is empty
    sql_str = ""
    try:
        sql_str = data.get("error_sql", {}).get("sql", "")
    except Exception:
        pass
    if not sql_str or not sql_str.strip():
        print("Discarding response with empty error SQL.")
        return None

    return data


# ─────────────────────────────────────────────
# Per-item async processor
# ─────────────────────────────────────────────
async def process_single_error(
    obj: dict,
    schema_info: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate one erroneous SQL for a given data item."""
    demo_example = get_demo_example()
    if not demo_example:
        return None
    prompt = generate_error_sql_prompt(obj, demo_example, schema_info)
    raw = await call_llm_async(prompt, semaphore)
    return parse_llm_response(raw)


async def process_data_item_async(
    obj: dict,
    semaphore: asyncio.Semaphore,
    max_error_sqls: int = 10,
) -> dict | None:
    """Process one data item: generate max_error_sqls erroneous SQLs concurrently."""
    db_id = obj.get("db_id")
    if not db_id:
        print(f"Skipping item without db_id: {obj}")
        return None

    schema_info = get_database_schema("SynSQL", db_id)

    # Fire all max_error_sqls requests concurrently
    tasks = [
        process_single_error(obj, schema_info, semaphore)
        for _ in range(max_error_sqls)
    ]
    results = await asyncio.gather(*tasks)
    error_results = [r for r in results if r is not None]

    if not error_results:
        return None

    return {
        "id": obj.get("id", ""),
        "question": obj.get("question"),
        "evidence": obj.get("external_knowledge", ""),
        "db_id": db_id,
        "gt_sql": obj.get("sql"),
        "schema": schema_info,
        "error_results": error_results,
    }


# ─────────────────────────────────────────────
# Main async pipeline
# ─────────────────────────────────────────────
async def main_async() -> None:
    print("Starting NL2SQL error data generation (async production mode) …")
    print(f"  Index range : [{start_index}, {end_index})")
    print(f"  MAX_CONCURRENCY : {MAX_CONCURRENCY}")

    preload_demo_examples()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    max_error_sqls = 10
    save_every = 5

    # Read all items in the target range first (ijson streaming)
    print("Loading data items from file …")
    items_to_process: list[dict] = []
    with open(DATA_FILE, "rb") as f:
        for obj_index, obj in enumerate(ijson.items(f, "item")):
            if obj_index < start_index:
                continue
            if obj_index >= end_index:
                break
            items_to_process.append(obj)

    print(f"Loaded {len(items_to_process)} items. Launching async tasks …")

    generated_data: list[dict] = []
    processed_count = 0

    # Process items with a progress bar; items themselves are sequential,
    # but within each item all LLM calls are concurrent.
    for obj in tqdm(items_to_process, desc="Processing items", colour="green"):
        result = await process_data_item_async(obj, semaphore, max_error_sqls)
        if result:
            generated_data.append(result)

        processed_count += 1
        if processed_count % save_every == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                json.dump(generated_data, out_f, indent=2, ensure_ascii=False)
            print(f"  [checkpoint] Saved {processed_count} items processed, "
                  f"{len(generated_data)} with results.")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(generated_data, out_f, indent=2, ensure_ascii=False)

    print(f"\nDone! Generated {len(generated_data)} error data records → {OUTPUT_FILE}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
