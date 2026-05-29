"""
DiagSQL - Dataset for RL Training

Loads the generated_error_data_2k.json and prepares it for GRPO training.
"""
import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data_types import MiniBatch

# Database paths
SYNSQL_SQLITE_BASE = Path("DiagSQL/dataset/OmniSQL")

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
    '- If the SQL is correct, output: {"label": "right"}\n'
    '- If the SQL is wrong, output: {\n'
    '        "label": "wrong",\n'
    '        "error_types": [\n'
    '            {\n'
    '                "reasoning": "explanation of why this error type applies",\n'
    '                "error_type": "error category from the above list",\n'
    '                "sub_error_type": "specific error type from the above list"\n'
    '            }\n'
    '        ]\n'
    '    }\n'
)

USER_TEMPLATE = (
    "Question: {question}\n"
    "Evidence: {evidence}\n"
    "Database Schema:\n{schema}\n"
    "Generated SQL: {sql}\n"
    "Execution Result: {execution_result}\n\n"
    "Please analyze whether this SQL query correctly answers the question based on the provided evidence, database schema, and execution result. "
    "If the SQL is incorrect, identify the specific error types."
)

# Process-level cache
_QUERY_CACHE: Dict = {}


def _execute_sql_query(db_id: str, sql: str, sqlite_base: Path, limit: int = 100) -> Dict[str, Any]:
    """Execute SQL query with caching."""
    if not db_id or not sql:
        return {"success": False, "error": "Missing db_id or sql"}

    sql_lower = sql.lower().strip()
    sql_exec = sql if "limit" in sql_lower else f"{sql.rstrip(';')} LIMIT {limit}"
    cache_key = (str(sqlite_base), db_id, sql_exec)
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    sqlite_path = sqlite_base / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        result = {"success": False, "error": f"Database not found: {sqlite_path}"}
        _QUERY_CACHE[cache_key] = result
        return result

    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=5)
        cursor = conn.cursor()
        cursor.execute(sql_exec)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        result = {
            "success": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    except Exception as e:
        result = {"success": False, "error": str(e)}

    _QUERY_CACHE[cache_key] = result
    return result


class DiagSQLDataset(Dataset):
    """
    Dataset for DiagSQL RL training.

    Loads generated_error_data_2k.json and expands each entry into:
    - 1 correct SQL sample (label=True)
    - N error SQL samples (label=False, with error_types)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if max_samples:
            raw_data = raw_data[:max_samples]

        # Expand to flat list
        self.samples: List[Dict[str, Any]] = []
        for item in raw_data:
            question = item['question']
            evidence = item.get('evidence', '')
            db_id = item['db_id']
            schema = item.get('schema', '')

            # Correct SQL as positive sample
            gt_sql = item.get('gt_sql', '')
            if gt_sql:
                self.samples.append({
                    'question': question,
                    'evidence': evidence,
                    'db_id': db_id,
                    'sql': gt_sql,
                    'label': True,
                    'error_types': [],
                    'schema': schema,
                })

            # Error SQLs as negative samples
            for err_result in item.get('error_results', []):
                err_sql_obj = err_result.get('error_sql', {})
                err_sql = err_sql_obj.get('sql', '')
                err_error_types = err_sql_obj.get('error_types', [])
                if err_sql:
                    self.samples.append({
                        'question': question,
                        'evidence': evidence,
                        'db_id': db_id,
                        'sql': err_sql,
                        'label': False,
                        'error_types': err_error_types,
                        'schema': schema,
                    })

        self.tokenizer = tokenizer
        self.sqlite_base = SYNSQL_SQLITE_BASE

        pos = sum(1 for s in self.samples if s['label'])
        neg = sum(1 for s in self.samples if not s['label'])
        print(f"[DiagSQLDataset] Loaded {len(raw_data)} raw entries -> {len(self.samples)} samples "
              f"(pos={pos}, neg={neg})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = dict(self.samples[idx])
        item.update(self._encode_prefix(
            item["question"],
            item["evidence"],
            item["db_id"],
            item["sql"],
            item["schema"],
        ))
        return item

    def _encode_prefix(
        self,
        question: str,
        evidence: str,
        db_id: str,
        sql: str,
        schema: str,
    ) -> Dict[str, Any]:
        """Encode the prompt prefix."""
        # Execute SQL to get result
        execution_result = _execute_sql_query(db_id, sql, self.sqlite_base)

        if execution_result["success"]:
            result_str = f"Success: {execution_result['row_count']} rows returned"
            if execution_result['rows']:
                result_str += f"\nSample data: {execution_result['rows'][:3]}"
        else:
            result_str = f"Error: {execution_result['error']}"

        user_message = USER_TEMPLATE.format(
            question=question,
            evidence=evidence,
            schema=schema,
            sql=sql,
            execution_result=result_str,
        )

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        try:
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prefix = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                prefix = f"System: {SYSTEM_MESSAGE}\n\nUser: {user_message}\n\nAssistant:"
        except Exception:
            prefix = f"System: {SYSTEM_MESSAGE}\n\nUser: {user_message}\n\nAssistant:"

        tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

        return {
            "prefix": prefix,
            "prefix_tokens": self.tokenizer.convert_ids_to_tokens(tokens),
            "prefix_token_ids": tokens,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        return MiniBatch(
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
            target=[1 if item["label"] else 0 for item in batch],
            question=[item["question"] for item in batch],
            evidence=[item.get("evidence", "") for item in batch],
            db_id=[item["db_id"] for item in batch],
            sql=[item["sql"] for item in batch],
            error_types=[item.get("error_types", []) for item in batch],
            numbers=[[] for _ in batch],
        )
