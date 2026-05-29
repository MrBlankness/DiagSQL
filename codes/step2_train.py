"""
DiagSQL Step 2: Model Training (SFT + GRPO)

两阶段训练：先 SFT warm-up，再 GRPO RL 训练。

Usage:
    python step2_train.py --config configs/config_test.yaml
    python step2_train.py --config configs/config_full.yaml
"""
import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fix_tokenizer_config(model_path: str):
    """Fix extra_special_tokens format for newer transformers."""
    tc_path = Path(model_path) / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path, 'r') as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            tc["extra_special_tokens"] = {tok: tok for tok in tc["extra_special_tokens"]}
            with open(tc_path, 'w') as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            logger.info(f"  Fixed tokenizer_config.json: {tc_path}")


def train_sft(cfg: dict):
    """Stage 1: SFT Training with LoRA."""
    logger.info("=" * 60)
    logger.info("  Step 2a: SFT Training")
    logger.info("=" * 60)

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(cfg["output_dir"])
    sft_dir = output_dir / "sft_data"
    sft_out = output_dir / "sft_checkpoint"
    merged_out = output_dir / "sft_merged"
    sft_out.mkdir(parents=True, exist_ok=True)

    base_model = cfg["base_model_path"]
    sft_cfg = cfg["step2"]["sft"]

    # Fix tokenizer
    fix_tokenizer_config(base_model)

    # Load tokenizer
    logger.info(f"  Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    def load_jsonl(path):
        samples = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    train_raw = load_jsonl(sft_dir / "sft_train.jsonl")
    val_raw = load_jsonl(sft_dir / "sft_val.jsonl")
    logger.info(f"  Data: {len(train_raw)} train, {len(val_raw)} val")

    def format_messages(sample):
        try:
            text = tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except:
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in sample["messages"]]
            text = "\n\n".join(parts)
        return {"text": text}

    train_dataset = Dataset.from_list(train_raw).map(format_messages, remove_columns=["messages", "source", "label"])
    val_dataset = Dataset.from_list(val_raw).map(format_messages, remove_columns=["messages", "source", "label"])

    # Load model
    logger.info(f"  Loading model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=sft_cfg["lora_r"],
        lora_alpha=sft_cfg["lora_alpha"],
        lora_dropout=sft_cfg["lora_dropout"],
        target_modules=sft_cfg["lora_target_modules"],
        bias="none", inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_kwargs = dict(
        output_dir=str(sft_out),
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=sft_cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        seed=42,
    )

    # Use max_steps or num_train_epochs
    if "max_steps" in sft_cfg:
        training_kwargs["max_steps"] = sft_cfg["max_steps"]
        training_kwargs["num_train_epochs"] = 1
    else:
        training_kwargs["num_train_epochs"] = sft_cfg.get("num_train_epochs", 3)
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = 100
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["save_steps"] = 100
        training_kwargs["save_total_limit"] = 3
        training_kwargs["load_best_model_at_end"] = True

    sft_config = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("  Training...")
    trainer.train()

    # Merge and save
    logger.info("  Merging LoRA and saving...")
    merged_out.mkdir(parents=True, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_out), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out))

    # Fix tokenizer in output
    fix_tokenizer_config(str(merged_out))

    logger.info(f"  ✓ SFT model saved to {merged_out}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return str(merged_out)


def train_grpo(cfg: dict, sft_model_path: str):
    """Stage 2: GRPO RL Training (FTRA + CORS)."""
    logger.info("=" * 60)
    logger.info("  Step 2b: GRPO Training (FTRA + CORS)")
    logger.info("=" * 60)

    output_dir = Path(cfg["output_dir"])
    grpo_cfg = cfg["step2"]["grpo"]

    # Build GRPO training config
    grpo_train_config = {
        "model_path": sft_model_path,
        "device": "cuda",
        "dtype": "bfloat16",
        "train_data_path": str(output_dir / "grpo_data.json"),
        "cooccurrence_data_path": cfg["cooccurrence_path"],
        "batch_size": grpo_cfg["num_questions_per_batch"] * grpo_cfg["num_answer_per_question"],
        "num_questions_per_batch": grpo_cfg["num_questions_per_batch"],
        "num_answer_per_question": grpo_cfg["num_answer_per_question"],
        "max_gen_len": grpo_cfg["max_gen_len"],
        "micro_batch_size": grpo_cfg["micro_batch_size"],
        "learning_rate": grpo_cfg["learning_rate"],
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "temperature": grpo_cfg["temperature"],
        "format_reward_weight": 0.1,
        "label_reward_weight": 1.0,
        "error_type_reward_weight": 1.0,
        "use_cors": grpo_cfg["use_cors"],
        "cors_lambda": grpo_cfg["cors_lambda"],
        "cors_lift_threshold": grpo_cfg["cors_lift_threshold"],
        "random_seed": 42,
        "max_steps": grpo_cfg["max_steps"],
        "ckpt_save_interval": grpo_cfg.get("ckpt_save_interval", grpo_cfg["max_steps"]),
        "eval_interval": grpo_cfg.get("ckpt_save_interval", grpo_cfg["max_steps"]),
        "skip_unfinished_episodes": False,
        "entropy_coeff": grpo_cfg["entropy_coeff"],
        "rollout_batch_size": grpo_cfg["rollout_batch_size"],
        "use_gradient_checkpointing": grpo_cfg["use_gradient_checkpointing"],
        "use_flash_attention": True,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "warmup_steps": 10,
        "ckpt_dir": str(output_dir / "grpo_checkpoint"),
        "log_dir": str(output_dir / "grpo_logs"),
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }

    config_path = output_dir / "grpo_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(grpo_train_config, f, default_flow_style=False)

    logger.info(f"  Config: {config_path}")
    logger.info(f"  Training GRPO ({grpo_cfg['max_steps']} steps)...")

    sys.path.insert(0, str(Path(__file__).parent))
    from train_diagsql import main as train_main
    train_main(str(config_path))

    # Find best checkpoint
    ckpt_dir = output_dir / "grpo_checkpoint"
    best_path = ckpt_dir / "ckpt_best"
    if not best_path.exists():
        # Fallback to last step
        step_dirs = sorted(ckpt_dir.glob("ckpt_step_*"))
        best_path = step_dirs[-1] if step_dirs else ckpt_dir

    # Fix tokenizer
    fix_tokenizer_config(str(best_path))

    logger.info(f"  ✓ GRPO model saved to {best_path}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    return str(best_path)


def main():
    parser = argparse.ArgumentParser(description="DiagSQL Step 2: Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT, use existing checkpoint")
    parser.add_argument("--sft_model_path", type=str, default=None, help="Path to existing SFT model")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.chdir(str(Path(__file__).parent))

    if args.skip_sft and args.sft_model_path:
        sft_model_path = args.sft_model_path
        logger.info(f"  Skipping SFT, using: {sft_model_path}")
    else:
        sft_model_path = train_sft(cfg)

    grpo_model_path = train_grpo(cfg, sft_model_path)

    # Save final model path for next steps
    info = {"sft_model": sft_model_path, "grpo_model": grpo_model_path}
    info_path = Path(cfg["output_dir"]) / "model_paths.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"  Model paths saved to {info_path}")
    logger.info("✓ Step 2 完成")


if __name__ == "__main__":
    main()
