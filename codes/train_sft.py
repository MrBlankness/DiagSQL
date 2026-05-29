"""
SFT Training Script for SQL Error Analysis
Supports: Qwen3-8B, Meta-Llama-3.1-8B-Instruct

Uses HuggingFace TRL SFTTrainer with LoRA (PEFT).
Trains on conversation-formatted data: system + user + assistant turns.

Usage:
    python train_sft.py --model qwen3   # Train Qwen3-8B
    python train_sft.py --model llama   # Train Llama-3.1-8B

Outputs saved to:
    checkpoints/qwen3-8b-sql-sft/
    checkpoints/llama-3.1-8b-sql-sft/
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("")
SFT_DATA_DIR = BASE / "sft" / "sft_data"
CKPT_DIR     = BASE / "sft" / "checkpoints"

MODEL_CONFIGS = {
    "qwen3": {
        "model_path": "Qwen3-8B",
        "output_dir": str(CKPT_DIR / "qwen3-8b-sql-sft"),
        # Qwen3 uses apply_chat_template with enable_thinking=False
        "enable_thinking": False,
        # LoRA target modules for Qwen3
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
    },
    "llama": {
        "model_path": "Meta-Llama-3.1-8B-Instruct",
        "output_dir": str(CKPT_DIR / "llama-3.1-8b-sql-sft"),
        "enable_thinking": False,
        # LoRA target modules for Llama
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
    },
}

# ── Hyperparameters ────────────────────────────────────────────────────────
LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05
MAX_SEQ_LENGTH    = 4096
TRAIN_BATCH_SIZE  = 2
GRAD_ACCUM_STEPS  = 8       # effective batch = 2 * 8 = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
WARMUP_RATIO      = 0.05
LR_SCHEDULER      = "cosine"
SAVE_STEPS        = 100
EVAL_STEPS        = 100
LOGGING_STEPS     = 20
FP16              = False
BF16              = True    # H20 supports bfloat16


# ── Data Loading ───────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def format_messages(sample, tokenizer, enable_thinking: bool = False):
    """
    Apply chat template to convert messages list → single text string.
    Returns {"text": rendered_string}.
    """
    messages = sample["messages"]
    try:
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        if enable_thinking:
            kwargs["enable_thinking"] = True
        else:
            # Suppress Qwen3 thinking tokens if present
            try:
                kwargs["enable_thinking"] = False
            except Exception:
                pass
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e}, using fallback")
        # Fallback: manual formatting
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        text = "\n\n".join(parts)
    return {"text": text}


# ── Main Training ──────────────────────────────────────────────────────────

def train(model_key: str, args):
    cfg = MODEL_CONFIGS[model_key]
    model_path  = cfg["model_path"]
    output_dir  = cfg["output_dir"]
    enable_thinking = cfg["enable_thinking"]
    lora_targets = cfg["lora_target_modules"]

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{'='*60}")
    logger.info(f"Model:      {model_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Train data: {SFT_DATA_DIR}/sft_train.jsonl")

    # ── Tokenizer ────────────────────────────────────────────────────────
    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Vocab size: {len(tokenizer)}, pad_token: {tokenizer.pad_token!r}")

    # ── Load Data ─────────────────────────────────────────────────────────
    logger.info("Loading SFT data ...")
    train_raw = load_jsonl(SFT_DATA_DIR / "sft_train.jsonl")
    val_raw   = load_jsonl(SFT_DATA_DIR / "sft_val.jsonl")
    logger.info(f"  Train: {len(train_raw)}, Val: {len(val_raw)}")

    def fmt(sample):
        return format_messages(sample, tokenizer, enable_thinking)

    train_dataset = Dataset.from_list(train_raw).map(fmt, remove_columns=["messages", "source", "label"])
    val_dataset   = Dataset.from_list(val_raw).map(fmt, remove_columns=["messages", "source", "label"])
    logger.info(f"  Dataset formatted. Sample text[:200]: {train_dataset[0]['text'][:200]!r}")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    logger.info(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── LoRA ──────────────────────────────────────────────────────────────
    logger.info(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=lora_targets,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training Args ─────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=BF16,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    logger.info("Saving final model ...")
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Model saved to {final_path}")

    # Save training config for reference
    config_info = {
        "model_key": model_key,
        "model_path": model_path,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(Path(output_dir) / "train_config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    logger.info(f"Training complete! Config saved to {output_dir}/train_config.json")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT training for SQL error analysis")
    parser.add_argument(
        "--model", choices=["qwen3", "llama"], required=True,
        help="Which model to train: qwen3 (Qwen3-8B) or llama (Meta-Llama-3.1-8B-Instruct)"
    )
    args = parser.parse_args()
    train(args.model, args)


if __name__ == "__main__":
    main()
