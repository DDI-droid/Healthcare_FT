# gspo_readmit/utils.py
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- basic utils -------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log(msg: str):
    print(f"[gspo] {msg}", flush=True)

# ---- prompt helpers ----------------------------------------------------------

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def format_prompt(system_text: str, user_tmpl: str, text: str, tokenizer=None) -> str:
    """Build the canonical prompt using chat template if available, otherwise fallback."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Use GPT-OSS chat template
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_tmpl.format(text=text)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        # Fallback to manual format if no chat template
        return f"<|system|>\n{system_text}\n<|user|>\n{user_tmpl.format(text=text)}\n<|assistant|>\n"

# ---- (quantized) model loading & LoRA merge ---------------------------------

def bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

@dataclass
class LoadSpec:
    base_model: str
    adapters_dir: Optional[str] = None   # where GSPO/PEFT adapters were saved
    use_4bit: bool = True                # load base in 4-bit (QLoRA inference)
    trust_remote_code: bool = True
    device_map: str = "auto"

def load_tokenizer(model_or_dir: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(model_or_dir, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base_model(spec: LoadSpec):
    quant = bnb_4bit_config() if spec.use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        trust_remote_code=spec.trust_remote_code,
        device_map=spec.device_map,
        quantization_config=quant,
    )
    return model

def attach_or_load_adapters(model, adapters_dir: Optional[str]):
    """Attach LoRA adapters if provided."""
    if not adapters_dir:
        return model
    # Try to load adapters from the directory where GSPO saved them
    model = PeftModel.from_pretrained(model, adapters_dir)
    return model

def prepare_lora_trainable(model, r=16, alpha=32, dropout=0.05):
    """If you need to put LoRA back for extra SFT steps."""
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)

def merge_lora_and_save(model, out_dir: str):
    """
    Merge LoRA adapters into base weights for portable inference
    (FP16 weights on disk). After merging, the PEFT wrapper is removed.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.merge_and_unload()   # PEFT utility
    model.save_pretrained(out_dir)
    return out_dir

def detect_saved_base_id(adapters_dir: str) -> Optional[str]:
    """
    If you stored a small JSON note during training containing the base model id,
    read it here; otherwise return None and require manual base id.
    """
    meta = os.path.join(adapters_dir, "gspo_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta, "r", encoding="utf-8") as f:
                j = json.load(f)
            return j.get("base_model")
        except Exception:
            return None
    return None
