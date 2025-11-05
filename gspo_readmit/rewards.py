# gspo_readmit/rewards.py
import re
from typing import List, Dict

YES = re.compile(r"\bYES\b", re.IGNORECASE)
NO  = re.compile(r"\bNO\b",  re.IGNORECASE)

def parse_final_answer(text: str) -> str | None:
    # look for "Final Answer: YES/NO" first; else fallback to last YES/NO
    m = re.search(r"Final\s*Answer\s*:\s*(YES|NO)\b", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    # fallback: last occurrence
    cands = list(re.finditer(r"\b(YES|NO)\b", text, re.IGNORECASE))
    return cands[-1].group(1).upper() if cands else None

def reward_batch(prompts: List[str], outputs: List[str], labels: List[int]) -> Dict[str, List[float]]:
    """Return per-sample rewards for GSPO (1.0 correct YES/NO, else 0.0), 
       plus small formatting penalties/bonuses.
       Also returns predictions for metrics computation."""
    rewards = []
    predictions = []
    for out, y in zip(outputs, labels):
        ans = parse_final_answer(out or "")
        base = 0.0
        pred = None
        if ans is not None:
            pred = 1 if ans == "YES" else 0
            base = 1.0 if pred == y else 0.0
        predictions.append(pred if pred is not None else -1)  # -1 for invalid predictions
        # small structure reward to keep format neat
        fmt_bonus = 0.1 if "Reasoning:" in (out or "") and "Final Answer:" in (out or "") else 0.0
        # brevity guard: discourage >2200 tokens dumps (scaled); we won't inspect the reasoning content
        length_pen = 0.0  # if needed, add from token counts via callback
        rewards.append(base + fmt_bonus - length_pen)
    return {
        "rewards": rewards,
        "predictions": predictions,  # For metrics computation
        "labels": labels,  # For metrics computation
    }
