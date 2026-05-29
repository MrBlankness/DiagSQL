"""
DiagSQL - GRPO Rollout and Policy Update

Implements the improved GRPO algorithm from DiagSQL with:
- FTRA (Fine-grained Token-level Reward Allocation)
- CORS (Co-occurrence-aware Reward Shaping)
- Group normalization for advantages
"""
import gc
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel

from data_types import Episode, MiniBatch
from cors import CooccurrenceMatrix
from ftra import compute_ftra_rewards, compute_reward_info


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: MiniBatch,
    max_gen_len: int,
    num_answer_per_question: int,
    device: torch.device,
    dtype: torch.dtype,
    cooccurrence_matrix: Optional[CooccurrenceMatrix] = None,
    cors_lambda: float = 0.3,
    cors_lift_threshold: float = 1.0,
    format_reward_weight: float = 0.1,
    rollout_batch_size: int = 2,
    temperature: float = 1.0,
) -> List[Episode]:
    """
    Execute rollout: generate responses and compute FTRA+CORS rewards.

    Args:
        model: Policy model
        tokenizer: Tokenizer
        batch: Input batch
        max_gen_len: Maximum generation length
        num_answer_per_question: Number of samples per question (G in paper)
        device: Device
        dtype: Data type
        cooccurrence_matrix: Pre-computed co-occurrence matrix for CORS
        cors_lambda: CORS lambda parameter
        cors_lift_threshold: CORS lift threshold
        format_reward_weight: Weight for format reward in FTRA
        rollout_batch_size: Number of questions per generation sub-batch
        temperature: Sampling temperature

    Returns:
        List of Episodes with FTRA+CORS token-level rewards
    """
    # Switch to eval mode for generation (needed for KV cache with gradient checkpointing)
    was_training = model.training
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    num_questions = len(batch.prefix)
    prefix_token_ids = batch.prefix_token_ids

    episodes: List[Episode] = []

    # Process in sub-batches to manage memory
    for q_start in range(0, num_questions, rollout_batch_size):
        q_end = min(q_start + rollout_batch_size, num_questions)
        sub_prefix_ids = prefix_token_ids[q_start:q_end]
        sub_size = q_end - q_start

        # Left-padding for batch generation
        max_prompt_len = max(len(t) for t in sub_prefix_ids)
        input_ids_list = []
        attn_mask_list = []
        for t in sub_prefix_ids:
            pad_len = max_prompt_len - len(t)
            input_ids_list.append([pad_token_id] * pad_len + list(t))
            attn_mask_list.append([0] * pad_len + [1] * len(t))

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attn_mask_tensor = torch.tensor(attn_mask_list, dtype=torch.long, device=device)

        # Generate
        output_ids = model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask_tensor,
            max_new_tokens=max_gen_len,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            num_return_sequences=num_answer_per_question,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        generated_ids_all = output_ids[:, max_prompt_len:].cpu()

        # Free GPU memory
        del input_ids_tensor, attn_mask_tensor, output_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Build episodes
        for i in range(sub_size):
            q_idx = q_start + i
            for j in range(num_answer_per_question):
                idx = i * num_answer_per_question + j
                gen_ids = generated_ids_all[idx].tolist()

                # Truncate to first pad/eos
                if pad_token_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(pad_token_id)]
                if eos_token_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_token_id) + 1]

                is_finished = (eos_token_id in gen_ids)
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Compute reward info
                expected_label = bool(batch.target[q_idx]) if batch.target[q_idx] is not None else None
                expected_error_types = batch.error_types[q_idx] if batch.error_types else []

                reward_info = compute_reward_info(
                    response=generated_text,
                    expected_label=expected_label,
                    expected_error_types=expected_error_types,
                )

                # Compute FTRA + CORS token-level rewards
                token_rewards, episode_reward = compute_ftra_rewards(
                    generated_text=generated_text,
                    reward_info=reward_info,
                    tokenizer=tokenizer,
                    cooccurrence_matrix=cooccurrence_matrix,
                    cors_lambda=cors_lambda,
                    cors_lift_threshold=cors_lift_threshold,
                    format_reward_weight=format_reward_weight,
                )

                episodes.append(Episode(
                    prefix=batch.prefix[q_idx],
                    text=batch.prefix[q_idx] + generated_text,
                    prefix_token_ids=batch.prefix_token_ids[q_idx],
                    prefix_tokens=batch.prefix_tokens[q_idx],
                    generated_token_ids=gen_ids,
                    is_finished=is_finished,
                    reward=episode_reward,
                    reward_info=reward_info,
                    token_rewards=token_rewards,
                    question=batch.question[q_idx],
                    evidence=batch.evidence[q_idx],
                    db_id=batch.db_id[q_idx],
                    sql=batch.sql[q_idx],
                    expected_label=expected_label,
                    expected_error_types=expected_error_types,
                ))

        del generated_ids_all
        gc.collect()

    # Log rollout stats
    finished = sum(ep.is_finished for ep in episodes)
    total_ep = len(episodes)
    mean_reward = np.mean([ep.reward for ep in episodes]) if episodes else 0.0
    label_correct = sum(1 for ep in episodes if ep.reward_info.get("label_correct", False))
    print(
        f"  Rollout: {total_ep} episodes | finished={finished} | "
        f"mean_reward={mean_reward:.3f} | label_correct={label_correct}/{total_ep}",
        flush=True,
    )

    # Restore training mode
    if was_training:
        model.train()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    Normalize rewards within each group (same prefix).

    GRPO computes advantage as: A_i = (R_i - μ_R) / σ_R
    Token-level rewards are scaled by the same ratio.
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix_token_ids)].append(episode)

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)

        for episode in group:
            if std_reward > 1e-8:
                normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-8)
            else:
                normalized_reward = 0.0

            # Scale token_rewards proportionally
            if episode.token_rewards is not None and abs(episode.reward) > 1e-8:
                scale = normalized_reward / episode.reward
                new_token_rewards = [r * scale for r in episode.token_rewards]
            elif episode.token_rewards is not None:
                new_token_rewards = [0.0] * len(episode.token_rewards)
            else:
                new_token_rewards = None

            output.append(Episode(
                prefix=episode.prefix,
                text=episode.text,
                prefix_token_ids=episode.prefix_token_ids,
                prefix_tokens=episode.prefix_tokens,
                generated_token_ids=episode.generated_token_ids,
                is_finished=episode.is_finished,
                reward=normalized_reward,
                reward_info=episode.reward_info,
                token_rewards=new_token_rewards,
                question=episode.question,
                evidence=episode.evidence,
                db_id=episode.db_id,
                sql=episode.sql,
                expected_label=episode.expected_label,
                expected_error_types=episode.expected_error_types,
            ))

    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of logits distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def update_policy(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    entropy_coeff: float = 0.01,
) -> Dict[str, float]:
    """
    Update policy using improved GRPO with token-level rewards.

    The GRPO objective with FTRA:
        J(θ) = E[Σ_t log π_θ(y_t | s, y_{<t}) * A_{i,t}]

    where A_{i,t} is the token-level advantage from FTRA+CORS.

    Args:
        model: Policy model
        optimizer: Optimizer
        episodes: Episodes with token-level rewards
        micro_batch_size: Micro batch size for gradient accumulation
        pad_token_id: Pad token id
        max_grad_norm: Max gradient norm for clipping
        device: Device
        dtype: Data type
        entropy_coeff: Entropy regularization coefficient

    Returns:
        Dict with loss, grad_norm, entropy metrics
    """
    # Normalize rewards per group (GRPO advantage computation)
    episodes = normalize_rewards_per_group(episodes)

    # Sort by length for efficient padding
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    total_loss = 0.0
    total_grad_norm = 0.0
    total_entropy = 0.0
    num_target_tokens = 0

    optimizer.zero_grad(set_to_none=True)

    for i in range(0, len(episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]

        # Prepare batch tensors
        batch_lengths = [len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in batch_episodes]
        batch_max_length = max(batch_lengths)

        batch_token_ids = []
        batch_masks = []
        batch_token_rewards = []

        for episode in batch_episodes:
            token_ids = episode.prefix_token_ids + episode.generated_token_ids
            mask = [0] * len(episode.prefix_token_ids) + [1] * len(episode.generated_token_ids)

            # Pad
            pad_len = batch_max_length - len(token_ids)
            token_ids += [pad_token_id] * pad_len
            mask += [0] * pad_len

            # Token rewards
            if episode.token_rewards is not None and len(episode.token_rewards) == len(episode.generated_token_ids):
                token_rewards = [0.0] * len(episode.prefix_token_ids) + episode.token_rewards
            else:
                # Fallback: uniform distribution
                uniform = episode.reward / max(len(episode.generated_token_ids), 1)
                token_rewards = [0.0] * len(episode.prefix_token_ids) + [uniform] * len(episode.generated_token_ids)

            token_rewards += [0.0] * pad_len

            batch_token_ids.append(token_ids)
            batch_masks.append(mask)
            batch_token_rewards.append(token_rewards)

        # Convert to tensors
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_token_rewards = torch.tensor(batch_token_rewards, device=device, dtype=torch.float32)

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=dtype):
            input_ids = batch_token_ids[:, :-1]
            target_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            target_rews = batch_token_rewards[:, 1:]

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Per-token log probability
            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none",
            ).reshape(input_ids.shape[0], -1)

        # Entropy regularization
        if entropy_coeff > 0.0:
            with torch.autocast(device_type=device.type, dtype=dtype):
                token_entropy = compute_entropy(logits)
            entropy_loss = -(entropy_coeff * (token_entropy * target_masks).sum()
                             / max(target_masks.sum().item(), 1))
            entropy_loss_scaled = entropy_loss / num_micro_batches
        else:
            entropy_loss_scaled = None

        with torch.no_grad():
            token_entropy_no_grad = compute_entropy(logits)
            total_entropy += (token_entropy_no_grad * target_masks).sum().item()
            num_target_tokens += target_masks.sum().item()

        # GRPO objective: log_prob * token_reward (weighted by advantage)
        obj = (log_probs * target_rews * target_masks).sum() / max(target_masks.sum().item(), 1)
        loss = -obj / num_micro_batches
        total_loss += loss.item() * num_micro_batches

        if entropy_loss_scaled is not None:
            (loss + entropy_loss_scaled).backward()
        else:
            loss.backward()

        # Free memory
        del batch_token_ids, batch_masks, batch_token_rewards
        del input_ids, target_ids, target_masks, target_rews
        del outputs, logits, log_probs, loss, obj
        if entropy_loss_scaled is not None:
            del token_entropy, entropy_loss, entropy_loss_scaled
        del token_entropy_no_grad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Gradient step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    total_grad_norm = grad_norm.item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / num_micro_batches
    avg_entropy = total_entropy / max(num_target_tokens, 1)

    return {
        "loss": avg_loss,
        "grad_norm": total_grad_norm,
        "entropy": avg_entropy,
    }
