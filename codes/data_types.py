"""
DiagSQL RL Training - Data Types
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class Episode:
    """Stores all relevant information for a single episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, Any]
    token_rewards: Optional[List[float]] = None
    question: Optional[str] = None
    evidence: Optional[str] = None
    db_id: Optional[str] = None
    sql: Optional[str] = None
    expected_label: Optional[bool] = None
    expected_error_types: Optional[List[Dict]] = None


@dataclass
class MiniBatch:
    """Data batch for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    target: List[int]
    question: List[str]
    evidence: List[str]
    db_id: List[str]
    sql: List[str]
    error_types: List[List[Dict]]
    numbers: List[List[int]] = None


@dataclass
class DiagSQLConfig:
    """DiagSQL RL Training Configuration."""

    # Model
    model_path: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"

    # GRPO Training
    batch_size: int = 8
    num_questions_per_batch: int = 4
    max_gen_len: int = 1024
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    micro_batch_size: int = 2

    # Data
    train_data_path: str = ""
    cooccurrence_data_path: str = ""

    # Training control
    random_seed: int = 42
    max_steps: int = 200
    ckpt_save_interval: int = 50
    eval_interval: int = 20
    skip_unfinished_episodes: bool = True

    # FTRA (Fine-grained Token-level Reward Allocation)
    format_reward_weight: float = 0.1
    label_reward_weight: float = 1.0
    error_type_reward_weight: float = 1.0

    # CORS (Co-occurrence-aware Reward Shaping)
    use_cors: bool = True
    cors_lambda: float = 0.3  # λ in Eq. 5
    cors_lift_threshold: float = 1.0  # |log2(lift)| threshold

    # Rollout
    rollout_batch_size: int = 2
    num_answer_per_question: int = 2
    temperature: float = 1.0

    # Optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    entropy_coeff: float = 0.01

    # Optimizer
    betas: List = None
    eps: float = 1e-8
    warmup_steps: int = 10

    # Logging
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Multi-GPU (DeepSpeed)
    use_deepspeed: bool = False

    # Dataloader
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
