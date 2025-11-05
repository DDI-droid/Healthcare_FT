from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from gspo_readmit.cfg import TrainCfg

def main():
    cfg = TrainCfg()
    checkpoint = cfg.save_dir
    
    # Load base model and tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto",
    )
    
    # Load PEFT adapters
    model = PeftModel.from_pretrained(base_model, checkpoint)
    
    # Merge and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{checkpoint}/merged")
    tok.save_pretrained(f"{checkpoint}/merged")
    print(f"Merged model saved to {checkpoint}/merged")

if __name__ == "__main__":
    main()

