# gspo_readmit/cfg.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainCfg:
    # Model
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"  # Dense 14B model with excellent reasoning
    trust_remote_code: bool = True
    max_input_tokens: int = 2048   # reduced from 4000 for memory
    max_new_tokens: int = 512      # reduced from 1024 for memory
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Quant (training) - Disabled 4-bit due to dtype issues, using bf16 + gradient checkpointing
    use_4bit: bool = False         # Disable 4-bit (dtype mismatch issues), use bf16 instead
    # GSPO - Heavily reduced for memory (bf16 without 4-bit uses more memory)
    gspo_rollouts: int = 4         # samples per input per step
    gspo_batch_size: int = 1       # process 1 prompt at a time
    gspo_accum: int = 4            # reduced to 4, effective batch = 1*4 = 4
    lr: float = 5e-5
    epochs: int = 5
    warmup_ratio: float = 0.03
    save_dir: str = "out/gspo-qwen25-readmit"
    data_path: str = "data/dataset.jsonl"  # path to JSONL training data file
    logging_steps: int = 10
    eval_steps: int = 100
    eval_split: float = 0.2  # fraction of data to use for validation (0.0 to disable)
    # WandB
    use_wandb: bool = True
    wandb_project: str = "gspo-readmit"
    wandb_run_name: Optional[str] = None  # None = auto-generate
    seed: int = 42
