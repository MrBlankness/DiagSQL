"""
Merge LoRA adapter weights into the base model.

Usage:
    python merge_lora.py
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "Meta-Llama-3.1-8B-Instruct"
LORA_CKPT_PATH   = "sft/checkpoints/llama-3.1-8b-sql-sft/final"
MERGED_SAVE_PATH = "sft/checkpoints/llama-3.1-8b-sql-sft/final_merged"


def main():
    Path(MERGED_SAVE_PATH).mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_CKPT_PATH, trust_remote_code=True
    )
    logger.info(f"  Vocab size: {len(tokenizer)}")

    # ── Load base model ───────────────────────────────────────────────────
    logger.info(f"Loading base model from {BASE_MODEL_PATH} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info(f"  Base model loaded. Params: {sum(p.numel() for p in base_model.parameters()) / 1e9:.2f}B")

    # ── Load LoRA adapter ─────────────────────────────────────────────────
    logger.info(f"Loading LoRA adapter from {LORA_CKPT_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_CKPT_PATH)
    logger.info("  LoRA adapter loaded.")

    # ── Merge ─────────────────────────────────────────────────────────────
    logger.info("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()
    logger.info("  Merge complete.")

    # ── Save ──────────────────────────────────────────────────────────────
    logger.info(f"Saving merged model to {MERGED_SAVE_PATH} ...")
    model.save_pretrained(MERGED_SAVE_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_SAVE_PATH)
    logger.info("Done! Merged model saved.")


if __name__ == "__main__":
    main()
