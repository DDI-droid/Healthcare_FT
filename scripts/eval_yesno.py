# scripts/eval_yesno.py
import re, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_final(text):
    m = re.search(r"Final\s*Answer\s*:\s*(YES|NO)\b", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.findall(r"\b(YES|NO)\b", text, re.IGNORECASE)
    return m[-1].upper() if m else None

def main():
    mod_dir = "out/gspo-gptoss-readmit"
    tok = AutoTokenizer.from_pretrained(mod_dir, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(mod_dir, trust_remote_code=True, device_map="auto")
    ds = Dataset.from_json("data/discharge.jsonl").train_test_split(test_size=0.2, seed=42)["test"]

    correct = 0
    for ex in ds:
        # Use chat template if available, otherwise fallback
        if hasattr(tok, 'apply_chat_template') and tok.chat_template is not None:
            messages = [
                {"role": "system", "content": "You are a careful clinical reasoning assistant.\nFormat:\nReasoning:\n...\nFinal Answer: YES or NO"},
                {"role": "user", "content": f"Discharge summary:\n{ex['text']}\n\nTask: Will this patient be readmitted within 30 days? Answer YES or NO."},
            ]
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = f"<|system|>\nYou are a careful clinical reasoning assistant.\nFormat:\nReasoning:\n...\nFinal Answer: YES or NO\n<|user|>\nDischarge summary:\n{ex['text']}\n\nTask: Will this patient be readmitted within 30 days? Answer YES or NO.\n<|assistant|>\n"
        out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=200, do_sample=False)
        txt = tok.decode(out[0], skip_special_tokens=True)
        ans = parse_final(txt)
        pred = 1 if ans == "YES" else 0 if ans == "NO" else None
        if pred is not None and pred == int(ex["label"]): correct += 1
    acc = correct/len(ds)
    print({"accuracy": acc})

if __name__ == "__main__":
    main()
