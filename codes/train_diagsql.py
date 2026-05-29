"""
DiagSQL - Main RL Training Script

Two-stage pipeline:
  Stage 1: SFT (already completed, checkpoint available)
  Stage 2: Improved GRPO with FTRA + CORS

Usage:
  # Single GPU (debug):
  python train_diagsql.py --config config_diagsql.yaml

  # Multi-GPU with accelerate (4 GPUs):
  accelerate launch --num_processes=4 train_diagsql.py --config config_diagsql.yaml
"""
import argparse
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_types import DiagSQLConfig, Episode
from dataset import DiagSQLDataset
from cors import CooccurrenceMatrix
from grpo import rollout, update_policy


def load_config(config_path: str) -> DiagSQLConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DiagSQLConfig(**config_dict)


def setup_model_and_tokenizer(config: DiagSQLConfig):
    """Setup model and tokenizer with multi-GPU support."""
    print(f"[Setup] Loading model from {config.model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DiagSQL training.")

    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"[Setup] {num_gpus} GPU(s) available")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    # Data type
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    # Tokenizer - handle extra_special_tokens compatibility issue
    import json as _json
    tokenizer_config_path = Path(config.model_path) / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, 'r') as _f:
            _tc = _json.load(_f)
        if isinstance(_tc.get("extra_special_tokens"), list):
            # Convert list to dict for newer transformers compatibility
            _tc["extra_special_tokens"] = {tok: tok for tok in _tc["extra_special_tokens"]}
            with open(tokenizer_config_path, 'w') as _f:
                _json.dump(_tc, _f, indent=2, ensure_ascii=False)
            print("[Setup] Patched tokenizer_config.json (extra_special_tokens list->dict)")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Attention implementation
    attn_impl = None
    if config.use_flash_attention:
        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
            print("[Setup] Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("[Setup] flash-attn not available, using SDPA")
    else:
        attn_impl = "sdpa"

    # Load model with device_map="auto" for multi-GPU
    model_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)

    # Gradient checkpointing for memory efficiency
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[Setup] Gradient checkpointing enabled")

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Setup] Model loaded. Total params: {total_params:,}, Trainable: {trainable_params:,}")

    return model, tokenizer, device, dtype


def setup_optimizer(config: DiagSQLConfig, model):
    """Setup optimizer."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=tuple(config.betas) if config.betas else (0.9, 0.999),
        eps=float(config.eps),
    )
    return optimizer


def save_checkpoint(model, tokenizer, output_dir: Path, tag: str = ""):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"[Checkpoint] Saved {tag} to {output_dir}")


def log_metrics(
    step: int,
    episodes: list,
    results: Dict[str, float],
    duration: float,
    tb_writer: SummaryWriter,
):
    """Log training metrics."""
    rewards = [ep.reward for ep in episodes]
    mean_reward = np.mean(rewards)
    label_correct_rate = np.mean([
        1.0 if ep.reward_info.get("label_correct", False) else 0.0
        for ep in episodes
    ])
    format_rate = np.mean([
        ep.reward_info.get("format_reward", 0.0)
        for ep in episodes
    ])
    mean_gen_len = np.mean([len(ep.generated_token_ids) for ep in episodes])

    print(
        f"\n  Step {step}: loss={results['loss']:.4f} "
        f"grad_norm={results['grad_norm']:.4f} "
        f"entropy={results['entropy']:.4f} "
        f"mean_reward={mean_reward:.4f} "
        f"label_acc={label_correct_rate:.4f} "
        f"format={format_rate:.4f} "
        f"gen_len={mean_gen_len:.0f} "
        f"dur={duration:.1f}s"
    )

    if tb_writer:
        tb_writer.add_scalar("train/loss", results["loss"], step)
        tb_writer.add_scalar("train/grad_norm", results["grad_norm"], step)
        tb_writer.add_scalar("train/entropy", results["entropy"], step)
        tb_writer.add_scalar("train/mean_reward", mean_reward, step)
        tb_writer.add_scalar("train/label_accuracy", label_correct_rate, step)
        tb_writer.add_scalar("train/format_rate", format_rate, step)
        tb_writer.add_scalar("train/mean_gen_len", mean_gen_len, step)
        tb_writer.add_scalar("train/duration", duration, step)


def main(config_path: str):
    """Main training function."""
    config = load_config(config_path)

    print("=" * 70)
    print("  DiagSQL RL Training (FTRA + CORS)")
    print("=" * 70)
    print(f"  Model: {config.model_path}")
    print(f"  Training data: {config.train_data_path}")
    print(f"  CORS enabled: {config.use_cors}")
    print(f"  CORS λ: {config.cors_lambda}")
    print(f"  Max steps: {config.max_steps}")
    print("=" * 70)

    # Set random seed
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)

    # Setup model
    model, tokenizer, device, dtype = setup_model_and_tokenizer(config)

    # Setup optimizer
    optimizer = setup_optimizer(config, model)

    # Setup co-occurrence matrix for CORS
    cooccurrence_matrix = None
    if config.use_cors:
        if config.cooccurrence_data_path and Path(config.cooccurrence_data_path).exists():
            cooccurrence_matrix = CooccurrenceMatrix(config.cooccurrence_data_path)
        else:
            # Compute from NL2SQL-Bugs dataset
            nl2sql_bugs_path = str(Path(config.train_data_path).parent / "NL2SQL-Bugs-with-evidence-enriched.json")
            cooccurrence_matrix = CooccurrenceMatrix()
            if Path(nl2sql_bugs_path).exists():
                cooccurrence_matrix.compute_from_dataset(nl2sql_bugs_path)
            else:
                print("[Warning] No co-occurrence data found. CORS will use zero matrix.")
                cooccurrence_matrix = None

    # Setup dataset and dataloader
    print(f"\n[Data] Loading training data from {config.train_data_path}")
    train_dataset = DiagSQLDataset(
        tokenizer=tokenizer,
        data_path=config.train_data_path,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.num_questions_per_batch,
        shuffle=True,
        collate_fn=DiagSQLDataset.collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        drop_last=True,
    )

    # Setup TensorBoard logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config.log_dir}/{current_time}")

    # Checkpoint directory
    ckpt_dir = Path(config.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n[Training] Starting GRPO training for {config.max_steps} steps...")
    step = 0
    best_reward = float('-inf')
    data_iter = iter(train_dataloader)

    while step < config.max_steps:
        step += 1
        start_time = time.time()

        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        print(f"\n{'='*50}")
        print(f"  Step {step}/{config.max_steps}")
        print(f"{'='*50}")

        # Rollout: generate and evaluate
        print("  [Rollout] Generating responses...")
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config.max_gen_len,
            num_answer_per_question=config.num_answer_per_question,
            device=device,
            dtype=dtype,
            cooccurrence_matrix=cooccurrence_matrix,
            cors_lambda=config.cors_lambda,
            cors_lift_threshold=config.cors_lift_threshold,
            format_reward_weight=config.format_reward_weight,
            rollout_batch_size=config.rollout_batch_size,
            temperature=config.temperature,
        )

        # Skip if no finished episodes
        if config.skip_unfinished_episodes:
            episodes = [ep for ep in episodes if ep.is_finished]

        if not episodes:
            print("  [Warning] No finished episodes. Skipping step.")
            continue

        # Policy update
        print("  [Update] Updating policy...")
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config.micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config.max_grad_norm,
            device=device,
            dtype=dtype,
            entropy_coeff=config.entropy_coeff,
        )

        # Timing
        duration = time.time() - start_time

        # Log metrics
        log_metrics(step, episodes, results, duration, tb_writer)

        # Track best
        mean_reward = np.mean([ep.reward for ep in episodes])
        if mean_reward > best_reward:
            best_reward = mean_reward
            save_checkpoint(model, tokenizer, ckpt_dir / "ckpt_best", tag="best")

        # Periodic checkpoint
        if step % config.ckpt_save_interval == 0:
            save_checkpoint(model, tokenizer, ckpt_dir / f"ckpt_step_{step:06d}", tag=f"step-{step}")

        # Print sample generation for monitoring
        if step % 10 == 1 and episodes:
            ep = episodes[0]
            gen_text = tokenizer.decode(ep.generated_token_ids, skip_special_tokens=True)
            print(f"\n  [Sample] Label correct: {ep.reward_info.get('label_correct', False)}")
            print(f"  [Sample] Response (first 300 chars): {gen_text[:300]}")

    # Save final checkpoint
    save_checkpoint(model, tokenizer, ckpt_dir / f"ckpt_step_{step:06d}", tag="final")

    tb_writer.close()
    print(f"\n[Done] Training completed. Best reward: {best_reward:.4f}")
    print(f"[Done] Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiagSQL RL Training")
    parser.add_argument("--config", type=str, default="config_diagsql.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)
