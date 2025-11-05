# scripts/sft_warmup.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from gspo_readmit.prompts import SYS, USER_TMPL
from gspo_readmit.cfg import TrainCfg
from gspo_readmit.trainer_gspo import get_bnb_4bit

def format_io(text, y, tokenizer=None):
    ans = "YES" if int(y)==1 else "NO"
    assistant_response = f"Reasoning:\n...\nFinal Answer: {ans}\n"
    
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER_TMPL.format(text=text)},
            {"role": "assistant", "content": assistant_response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback to manual format
        return f"<|system|>\n{SYS}\n<|user|>\n{USER_TMPL.format(text=text)}\n<|assistant|>\n{assistant_response}"

def main():
    cfg = TrainCfg()
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = Dataset.from_json("data/discharge.jsonl")
    ds = ds.map(lambda ex: {"text": format_io(ex["text"], ex["label"], tokenizer=tok)})
    bnb = get_bnb_4bit()
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, trust_remote_code=True, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    args = TrainingArguments(output_dir=cfg.save_dir+"/sft", per_device_train_batch_size=2, gradient_accumulation_steps=16, num_train_epochs=0.3, learning_rate=2e-4, bf16=True, logging_steps=10, report_to=["none"])
    tr = Trainer(model=model, tokenizer=tok, args=args, train_dataset=ds)
    tr.train()
    tr.save_model(cfg.save_dir+"/sft")
    tok.save_pretrained(cfg.save_dir+"/sft")

if __name__ == "__main__":
    main()
