# scripts/gspo_train.py
from gspo_readmit.cfg import TrainCfg
from gspo_readmit.data import get_dataset
from gspo_readmit.rewards import reward_batch
from gspo_readmit.trainer_gspo import prepare_model, build_gspo_trainer
from gspo_readmit.utils import set_seed

def main():
    cfg = TrainCfg()
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Load and validate dataset
    ds = get_dataset(cfg.data_path)
    print(f"Loaded dataset from {cfg.data_path} with {len(ds)} examples")
    
    # Prepare model and trainer
    model, tok = prepare_model(cfg)
    trainer = build_gspo_trainer(model, tok, ds, cfg, reward_batch)
    
    # Train
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(cfg.save_dir)
    tok.save_pretrained(cfg.save_dir)
    
    # Finish WandB run if enabled
    if cfg.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

if __name__ == "__main__":
    main()
