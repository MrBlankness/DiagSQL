"""
DiagSQL - Fine-grained Token-level Reward Allocation (FTRA)

Implements Section 3.2 of the paper:
  1. Parse Yi into non-overlapping spans: S_i = {s_fmt, s_label, s_err^(1), ..., s_err^(K)}
  2. Attach span-specific scalar reward to each span
  3. Broadcast span rewards back to constituent tokens

Token-level reward: r_{i,t} = Σ_{s ∈ S_i} r_s * I(y_{i,t} ∈ s)
"""
import json
import re
import bisect
from typing import Dict, List, Any, Optional, Tuple, Set

from transformers import PreTrainedTokenizer

from cors import (
    CooccurrenceMatrix,
    compute_cors_rewards,
    ALL_SUB_ERROR_TYPES,
)


def clean_response(response: str) -> str:
    """Clean LLM response text, removing think tags."""
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return cleaned.strip()


def try_recover_truncated_json(s: str) -> Optional[Dict]:
    """Try to recover truncated JSON response."""
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    label_m = re.search(r'"label"\s*:\s*"(right|wrong)"', s)
    if not label_m:
        return None

    label = label_m.group(1)
    if label == "right":
        return {"label": "right"}

    # label == "wrong": try to extract complete error_type objects
    et_match = re.search(r'"error_types"\s*:\s*\[', s)
    if not et_match:
        return {"label": "wrong", "error_types": []}

    et_start = et_match.end()
    error_types = []
    i = et_start
    while i < len(s):
        obj_start = s.find("{", i)
        if obj_start == -1:
            break
        depth = 0
        obj_end = -1
        for k in range(obj_start, len(s)):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    obj_end = k + 1
                    break
        if obj_end == -1:
            break
        try:
            obj = json.loads(s[obj_start:obj_end])
            if isinstance(obj, dict) and "error_type" in obj and "sub_error_type" in obj:
                error_types.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
        i = obj_end

    return {"label": "wrong", "error_types": error_types}


def parse_response(response: str) -> Optional[Dict]:
    """Parse model response into structured dict."""
    response = clean_response(response)
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return try_recover_truncated_json(response)


def _build_token_char_spans(
    token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Build character-level spans for each token.
    Returns (starts, ends): position of each token in the decoded string.
    """
    starts: List[int] = []
    ends: List[int] = []
    pos = 0
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        starts.append(pos)
        pos += len(tok_str)
        ends.append(pos)
    return starts, ends


def _tokens_in_span(
    span_start: int,
    span_end: int,
    tok_starts: List[int],
    tok_ends: List[int],
) -> List[int]:
    """
    Return token indices that overlap with [span_start, span_end) by > 50%.
    Uses binary search for efficiency.
    """
    lo = bisect.bisect_right(tok_ends, span_start)
    hi = bisect.bisect_left(tok_starts, span_end)

    indices = []
    for i in range(lo, hi):
        tok_len = tok_ends[i] - tok_starts[i]
        if tok_len == 0:
            continue
        overlap = max(0, min(tok_ends[i], span_end) - max(tok_starts[i], span_start))
        if overlap > tok_len * 0.5:
            indices.append(i)
    return indices


def compute_ftra_rewards(
    generated_text: str,
    reward_info: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cooccurrence_matrix: Optional[CooccurrenceMatrix] = None,
    cors_lambda: float = 0.3,
    cors_lift_threshold: float = 1.0,
    format_reward_weight: float = 0.1,
) -> Tuple[List[float], float]:
    """
    Compute Fine-grained Token-level Reward Allocation (FTRA) with CORS.

    This implements the core DiagSQL algorithm:
    1. Parse response into spans (format, label, error_types)
    2. Compute span-specific rewards:
       - r_fmt: binary format compliance
       - r_label: binary label correctness
       - r_err^(k): F1-based + CORS adjusted reward for each error type
    3. Allocate rewards to tokens in their respective spans

    Args:
        generated_text: Model generated text
        reward_info: Dict with evaluation results
        tokenizer: Tokenizer for token-span mapping
        cooccurrence_matrix: Pre-computed co-occurrence matrix for CORS
        cors_lambda: λ parameter for CORS
        cors_lift_threshold: Minimum |log2(lift)| to consider
        format_reward_weight: Weight for format reward

    Returns:
        (token_rewards, episode_reward):
            token_rewards: List[float] of per-token rewards
            episode_reward: Scalar episode-level reward
    """
    tokens = tokenizer(generated_text, add_special_tokens=False)
    token_ids = tokens["input_ids"]
    n = len(token_ids)
    if n == 0:
        return [], 0.0

    token_rewards = [0.0] * n

    format_reward = reward_info.get("format_reward", 0.0)
    analysis_reward = reward_info.get("analysis_reward", 0.0)
    label_correct = reward_info.get("label_correct", False)
    predicted_response = reward_info.get("predicted_response", {}) or {}
    expected_error_types = reward_info.get("expected_error_types", []) or []
    expected_label = reward_info.get("expected_label", None)

    # Pre-compute token character spans
    tok_starts, tok_ends = _build_token_char_spans(token_ids, tokenizer)

    # ════════════════════════════════════════════════════════════════════
    # Phase 1: Format Reward → uniformly distributed to all tokens
    # ════════════════════════════════════════════════════════════════════
    if format_reward == 1.0:
        fmt_per_token = format_reward_weight / n
        for i in range(n):
            token_rewards[i] += fmt_per_token

    # ════════════════════════════════════════════════════════════════════
    # Phase 2: Label Reward → allocated to "label" field tokens
    # ════════════════════════════════════════════════════════════════════
    label_value = predicted_response.get("label", "")
    if label_value and analysis_reward != 0.0:
        label_text = f'"label": "{label_value}"'
        label_start = generated_text.find(label_text)
        if label_start != -1:
            label_end = label_start + len(label_text)
            label_indices = _tokens_in_span(label_start, label_end, tok_starts, tok_ends)
            if label_indices:
                per_tok = analysis_reward / len(label_indices)
                for idx in label_indices:
                    token_rewards[idx] += per_tok

    # ════════════════════════════════════════════════════════════════════
    # Phase 3: Error Type Reward → FTRA + CORS
    # Each predicted error gets r_err = r_base + Δ_cooccur
    # Reward is allocated to the tokens of that specific error span
    # ════════════════════════════════════════════════════════════════════
    pred_error_types = predicted_response.get("error_types", []) if label_value == "wrong" else []

    if pred_error_types and (expected_label is False or expected_label == 0):
        # Compute CORS-adjusted rewards for each predicted error
        if cooccurrence_matrix is not None:
            cors_rewards = compute_cors_rewards(
                pred_error_types=pred_error_types,
                gt_error_types=expected_error_types,
                cooccurrence_matrix=cooccurrence_matrix,
                lambda_=cors_lambda,
                lift_threshold=cors_lift_threshold,
            )
        else:
            # Without CORS, use simple binary reward (base only)
            gt_subs = set()
            for et in expected_error_types:
                sub = et.get("sub_error_type", "").strip()
                if sub:
                    gt_subs.add(sub)

            cors_rewards = {}
            for et in pred_error_types:
                error_type = et.get("error_type", "").strip()
                sub_error_type = et.get("sub_error_type", "").strip()
                if sub_error_type:
                    key = (error_type, sub_error_type)
                    cors_rewards[key] = 1.0 if sub_error_type in gt_subs else 0.0

        # Allocate error type rewards to corresponding token spans
        for et in pred_error_types:
            error_type = et.get("error_type", "").strip()
            sub_error_type = et.get("sub_error_type", "").strip()
            if not sub_error_type:
                continue

            key = (error_type, sub_error_type)
            err_reward = cors_rewards.get(key, 0.0)

            if err_reward == 0.0:
                # For hallucinated errors (not in GT), apply negative reward
                gt_subs = {e.get("sub_error_type", "").strip() for e in expected_error_types}
                if sub_error_type not in gt_subs:
                    err_reward = -0.5  # Penalty for hallucination

            # Find the token span for this error object in the generated text
            # Look for the error_type and sub_error_type strings
            et_text = f'"error_type": "{error_type}"'
            set_text = f'"sub_error_type": "{sub_error_type}"'
            et_start = generated_text.find(et_text)
            set_start = generated_text.find(set_text)

            if et_start != -1 and set_start != -1:
                # Find the enclosing { } for this error object
                obj_start = generated_text.rfind("{", 0, et_start)
                obj_end = generated_text.find("}", set_start) + 1
                if obj_start != -1 and obj_end > obj_start:
                    err_indices = _tokens_in_span(obj_start, obj_end, tok_starts, tok_ends)
                    if err_indices:
                        per_tok = err_reward / len(err_indices)
                        for idx in err_indices:
                            token_rewards[idx] += per_tok

    # Also add penalty for missing errors (cascading penalty from CORS)
    # If gt has errors that are not predicted, we penalize label tokens
    if (expected_label is False or expected_label == 0) and expected_error_types:
        gt_subs = {e.get("sub_error_type", "").strip() for e in expected_error_types}
        pred_subs = {e.get("sub_error_type", "").strip() for e in pred_error_types}
        missed_errors = gt_subs - pred_subs

        if missed_errors and cooccurrence_matrix is not None:
            # Add a small penalty to the overall response for missing errors
            # This is distributed to the closing tokens of the error_types array
            miss_penalty = 0.0
            for missed in missed_errors:
                for pred_sub in pred_subs:
                    m_uv = cooccurrence_matrix.get_lift(pred_sub, missed)
                    if abs(m_uv) >= cors_lift_threshold:
                        miss_penalty -= cors_lambda * abs(m_uv) * 0.1

            if miss_penalty != 0.0 and n > 0:
                # Distribute to last few tokens (closing of JSON)
                last_tokens = min(5, n)
                per_tok = miss_penalty / last_tokens
                for idx in range(n - last_tokens, n):
                    token_rewards[idx] += per_tok

    # Compute episode-level reward (sum of all token rewards)
    episode_reward = sum(token_rewards)

    return token_rewards, episode_reward


def compute_reward_info(
    response: str,
    expected_label: Optional[bool],
    expected_error_types: Optional[List[Dict]],
) -> Dict[str, Any]:
    """
    Compute reward information for a generated response.

    Returns a dict with:
        - format_reward: 1.0 if valid JSON format, else 0.0
        - analysis_reward: +1.0 if label correct, -1.0 otherwise
        - label_correct: bool
        - predicted_response: parsed dict
        - expected_label: the ground truth
        - expected_error_types: the ground truth error types
    """
    response = clean_response(response)
    parsed = parse_response(response)

    if parsed is None:
        return {
            "format_reward": 0.0,
            "analysis_reward": -1.0,
            "label_correct": False,
            "predicted_response": None,
            "expected_label": expected_label,
            "expected_error_types": expected_error_types,
            "is_truncated": True,
        }

    # Format check
    format_reward = _check_format(parsed)

    # Label correctness
    label_correct = False
    pred_label = parsed.get("label", "")
    if (expected_label is True or expected_label == 1) and pred_label == "right":
        label_correct = True
    elif (expected_label is False or expected_label == 0) and pred_label == "wrong":
        label_correct = True

    analysis_reward = 1.0 if label_correct else -1.0

    return {
        "format_reward": format_reward,
        "analysis_reward": analysis_reward,
        "label_correct": label_correct,
        "predicted_response": parsed,
        "expected_label": expected_label,
        "expected_error_types": expected_error_types,
        "is_truncated": False,
    }


def _check_format(parsed: Dict) -> float:
    """Check if parsed response follows the correct JSON format."""
    if "label" not in parsed:
        return 0.0

    label = parsed["label"]
    if label == "right":
        return 1.0
    elif label == "wrong":
        if "error_types" not in parsed:
            return 0.0
        error_types = parsed["error_types"]
        if not isinstance(error_types, list) or not error_types:
            return 0.0
        for error in error_types:
            if not isinstance(error, dict):
                return 0.0
            if "error_type" not in error or "sub_error_type" not in error:
                return 0.0
        return 1.0
    return 0.0
