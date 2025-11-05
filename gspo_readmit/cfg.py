# gspo_readmit/cfg.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainCfg:
    # Model
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"  # Dense 14B model with excellent reasoning
    trust_remote_code: bool = True
    max_input_tokens: int = 2048   # shorter for speed
    max_new_tokens: int = 512      # shorter for speed
    # LoRA - reduced rank for faster training
    lora_r: int = 8             # reduced from 16 for speed
    lora_alpha: int = 16        # scale with r
    lora_dropout: float = 0.05
    # Quant (training) - Disabled 4-bit due to dtype issues, using bf16 + gradient checkpointing
    use_4bit: bool = False         # Disable 4-bit (dtype mismatch issues), use bf16 instead
    # GSPO - Optimized for speed and memory
    gspo_rollouts: int = 4         # reduced from 4 for speed (2x faster generation)
    gspo_batch_size: int = 4       # increased from 1 for better GPU utilization
    gspo_accum: int = 4            # must be divisible by gspo_rollouts, effective batch = 2*4 = 8
    lr: float = 5e-5
    epochs: int = 5
    warmup_ratio: float = 0.03
    save_dir: str = "out/gspo-qwen25-readmit"
    data_path: str = "data/dataset.jsonl"  # path to JSONL training data file
    logging_steps: int = 10
    eval_steps: int = 20
    eval_split: float = 0.2  # smaller validation set (10% instead of 20%) for speed
    # WandB
    use_wandb: bool = True
    wandb_project: str = "gspo-readmit"
    wandb_run_name: Optional[str] = None  # None = auto-generate
    seed: int = 42
