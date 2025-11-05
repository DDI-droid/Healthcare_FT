# gspo_readmit/inference.py
from __future__ import annotations
import re
from typing import List, Dict, Optional

import torch
from transformers import GenerationConfig

from .prompts import SYS as SYSTEM_PROMPT, USER_TMPL
from .utils import (
    LoadSpec,
    load_tokenizer,
    load_base_model,
    attach_or_load_adapters,
    format_prompt,
    log,
)

FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(YES|NO)\b", re.IGNORECASE)

def parse_final_answer(text: str) -> Optional[str]:
    m = FINAL_RE.search(text or "")
    if m: 
        return m.group(1).upper()
    # fallback: last standalone YES/NO in the output
    m2 = re.findall(r"\b(YES|NO)\b", text or "", flags=re.IGNORECASE)
    return m2[-1].upper() if m2 else None

class ReadmitInferencer:
    """
    Usage:
        spec = LoadSpec(base_model="openai/gpt-oss-20b",
                        adapters_dir="out/gspo-gptoss-readmit",
                        use_4bit=True)
        inf = ReadmitInferencer(spec)
        outs = inf.predict(["...summary 1...", "...summary 2..."])
    """
    def __init__(self, spec: LoadSpec):
        self.spec = spec
        self.tok = load_tokenizer(spec.adapters_dir or spec.base_model, spec.trust_remote_code)
        self.model = load_base_model(spec)
        self.model = attach_or_load_adapters(self.model, spec.adapters_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.0,     # 0.0 -> greedy (deterministic)
        top_p: float = 1.0,
        do_sample: bool = False,
        batch_size: int = 1,
    ) -> List[Dict]:
        outs: List[Dict] = []
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompts = [
                format_prompt(SYSTEM_PROMPT, USER_TMPL, x, tokenizer=self.tok) for x in batch
            ]
            toks = self.tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,  # model-dependent context; adjust if needed
            ).to(self.model.device)

            gen = self.model.generate(**toks, generation_config=gen_cfg)
            # Note: gen includes prompt+completion; decode fully then slice if you prefer
            for j, g in enumerate(gen):
                text = self.tok.decode(g, skip_special_tokens=True)
                final = parse_final_answer(text)
                outs.append({
                    "input": batch[j],
                    "output": text,
                    "final_answer": final,        # "YES" / "NO" / None
                    "pred_label": 1 if final == "YES" else (0 if final == "NO" else None),
                })
        return outs

if __name__ == "__main__":
    # quick CLI demo
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model id/path, e.g. openai/gpt-oss-20b")
    ap.add_argument("--adapters", required=True, help="Directory with GSPO LoRA adapters (trainer save_dir)")
    ap.add_argument("--fourbit", action="store_true", help="Load base in 4-bit (QLoRA) for inference")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    spec = LoadSpec(
        base_model=args.base,
        adapters_dir=args.adapters,
        use_4bit=args.fourbit,
    )
    inf = ReadmitInferencer(spec)

    log("Type/paste a discharge summary. Press Ctrl-D (Unix) / Ctrl-Z+Enter (Windows) to run.")
    text = sys.stdin.read()
    if not text.strip():
        log("No text provided.")
        sys.exit(0)

    outs = inf.predict(
        [text],
        max_new_tokens=args.max_new,
        temperature=(args.temperature if args.sample else 0.0),
        top_p=(args.top_p if args.sample else 1.0),
        do_sample=args.sample,
        batch_size=1,
    )
    print(outs[0])
