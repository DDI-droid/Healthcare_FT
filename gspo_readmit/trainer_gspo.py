# gspo_readmit/trainer_gspo.py
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Try GSPO first, fallback to GRPO if not available
try:
    from trl import GSPOConfig, GSPOTrainer
    USE_GRPO = False
except ImportError:
    from trl import GRPOConfig as GSPOConfig, GRPOTrainer as GSPOTrainer
    USE_GRPO = True
    print("GSPO not available, using GRPO instead (Group Relative Policy Optimization)")
from .prompts import SYS, USER_TMPL
from .metrics import compute_classification_metrics

try:
    import wandb
except ImportError:
    wandb = None

def get_bnb_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

def prepare_model(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # Load model with optional 4-bit quantization
    import torch
    quant_config = get_bnb_4bit() if cfg.use_4bit else None
    
    if cfg.use_4bit:
        print(f"Loading {cfg.base_model} with 4-bit quantization (QLoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=cfg.trust_remote_code,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,  # Ensure compute dtype is bf16
            device_map="auto",
        )
    else:
        print(f"Loading {cfg.base_model} in bf16")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    print(f"Model config dtype: {model.config.torch_dtype if hasattr(model.config, 'torch_dtype') else 'default'}")
    
    # Prepare for kbit training first (before LoRA)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok

def format_prompt(text: str, tokenizer=None) -> str:
    """Format prompt using chat template if available, otherwise fallback to manual format."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Use GPT-OSS chat template
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER_TMPL.format(text=text)},
        ]
        # apply_chat_template returns formatted string when not tokenizing
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        # Fallback to manual format if no chat template
        return f"<|system|>\n{SYS}\n<|user|>\n{USER_TMPL.format(text=text)}\n<|assistant|>\n"

def build_gspo_trainer(model, tok, ds, cfg, reward_fn):
    # Initialize WandB if enabled
    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "base_model": cfg.base_model,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "gspo_rollouts": cfg.gspo_rollouts,
                "gspo_batch_size": cfg.gspo_batch_size,
                "gspo_accum": cfg.gspo_accum,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "eval_split": cfg.eval_split,
                "max_input_tokens": cfg.max_input_tokens,
                "max_new_tokens": cfg.max_new_tokens,
            },
        )
    
    # Split dataset into train/validation if eval_split > 0
    if cfg.eval_split > 0:
        from datasets import Dataset
        split = ds.train_test_split(test_size=cfg.eval_split, seed=cfg.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = None
    
    # Build prompt list and label list using chat template for training
    # Validate that dataset is not empty after split
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty after split. Check eval_split value.")
    
    train_prompts = [format_prompt(ex["text"], tokenizer=tok) for ex in train_ds]
    train_labels  = [int(ex["label"]) for ex in train_ds]
    
    # Check for duplicate prompts (shouldn't happen but could cause issues)
    if len(train_prompts) != len(set(train_prompts)):
        print(f"Warning: {len(train_prompts) - len(set(train_prompts))} duplicate prompts found in training set")
    
    # Create mapping from prompt to label for correct alignment when GSPO shuffles/batches
    prompt_to_label = {prompt: label for prompt, label in zip(train_prompts, train_labels)}
    
    # Build validation prompts and labels if eval dataset exists
    eval_prompts = None
    eval_prompt_to_label = None
    if eval_ds is not None:
        eval_prompts = [format_prompt(ex["text"], tokenizer=tok) for ex in eval_ds]
        eval_labels = [int(ex["label"]) for ex in eval_ds]
        eval_prompt_to_label = {prompt: label for prompt, label in zip(eval_prompts, eval_labels)}
    
    # Format datasets for GRPOTrainer vs GSPOTrainer
    # GRPOTrainer expects list of dicts with "prompt" key, GSPOTrainer expects list of strings
    if USE_GRPO:
        train_dataset_formatted = [{"prompt": p} for p in train_prompts]
        eval_dataset_formatted = [{"prompt": p} for p in eval_prompts] if eval_prompts else None
    else:
        train_dataset_formatted = train_prompts
        eval_dataset_formatted = eval_prompts
    
    # Track predictions and labels for metrics
    train_predictions = []
    train_labels_for_metrics = []
    eval_predictions = []
    eval_labels_for_metrics = []

    # GRPOConfig doesn't support evaluation_strategy, so we build config based on what's available
    # Check if we're using GRPO (fallback) vs GSPO
    if USE_GRPO:
        # GRPOConfig parameters - remove evaluation_strategy and eval_steps
        gspo_cfg = GSPOConfig(
            output_dir=cfg.save_dir,
            per_device_train_batch_size=cfg.gspo_batch_size,
            gradient_accumulation_steps=cfg.gspo_accum,
            learning_rate=cfg.lr,
            num_train_epochs=cfg.epochs,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.eval_steps,
            max_prompt_length=cfg.max_input_tokens,
            max_completion_length=cfg.max_new_tokens,
            num_generations=cfg.gspo_rollouts,
            report_to=["wandb"] if cfg.use_wandb else ["none"],
            seed=cfg.seed,
        )
    else:
        # GSPOConfig supports evaluation_strategy
        gspo_cfg = GSPOConfig(
            output_dir=cfg.save_dir,
            per_device_train_batch_size=cfg.gspo_batch_size,
            gradient_accumulation_steps=cfg.gspo_accum,
            learning_rate=cfg.lr,
            num_train_epochs=cfg.epochs,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.eval_steps,
            max_prompt_length=cfg.max_input_tokens,
            max_completion_length=cfg.max_new_tokens,
            num_generations=cfg.gspo_rollouts,
            evaluation_strategy="steps" if cfg.eval_split > 0 else "no",
            eval_steps=cfg.eval_steps if cfg.eval_split > 0 else None,
            report_to=["wandb"] if cfg.use_wandb else ["none"],
            seed=cfg.seed,
        )

    def rf(samples, outputs):
        # Determine which label mapping to use: check if sample is in eval set first
        # If eval_prompt_to_label exists, check if sample is there; otherwise use train mapping
        labels_for_samples = []
        is_eval = False
        for sample in samples:
            if eval_prompt_to_label is not None and sample in eval_prompt_to_label:
                labels_for_samples.append(eval_prompt_to_label[sample])
                is_eval = True
            elif sample in prompt_to_label:
                labels_for_samples.append(prompt_to_label[sample])
            else:
                # Fallback (shouldn't happen): try to find in either mapping
                if eval_prompt_to_label is not None:
                    labels_for_samples.append(eval_prompt_to_label.get(sample, prompt_to_label.get(sample, 0)))
                else:
                    labels_for_samples.append(prompt_to_label.get(sample, 0))
        
        # Handle multiple generations per prompt (if num_generations > 1)
        # GSPO may flatten outputs: if we have N samples and K rollouts, outputs length = N*K
        # Match labels length to outputs length
        if len(outputs) == len(samples):
            # 1-to-1 mapping (num_generations = 1)
            labels_for_outputs = labels_for_samples
        elif len(outputs) > len(samples) and len(outputs) % len(samples) == 0:
            # Multiple outputs per sample - repeat labels accordingly
            num_rollouts = len(outputs) // len(samples)
            labels_for_outputs = [label for label in labels_for_samples for _ in range(num_rollouts)]
        else:
            # Fallback: repeat last label or pad (shouldn't happen in normal GSPO usage)
            labels_for_outputs = labels_for_samples * (len(outputs) // len(samples) + 1)
            labels_for_outputs = labels_for_outputs[:len(outputs)]
        
        # Get rewards and predictions from reward function
        reward_result = reward_fn(prompts=samples, outputs=outputs, labels=labels_for_outputs)
        
        # Store predictions for metrics (filter out invalid predictions)
        predictions = [p for p in reward_result.get("predictions", []) if p != -1]
        labels_valid = [l for p, l in zip(reward_result.get("predictions", []), labels_for_outputs) if p != -1]
        
        if is_eval:
            eval_predictions.extend(predictions)
            eval_labels_for_metrics.extend(labels_valid)
        else:
            train_predictions.extend(predictions)
            train_labels_for_metrics.extend(labels_valid)
        
        return reward_result

    # Create a callback to log metrics
    class MetricsCallback(TrainerCallback):
        """Callback to compute and log classification metrics."""
        def __init__(self, train_preds, train_labels, eval_preds, eval_labels):
            self.train_predictions = train_preds
            self.train_labels = train_labels
            self.eval_predictions = eval_preds
            self.eval_labels = eval_labels
            self.step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called during logging to compute and log metrics."""
            if logs is None:
                return
            
            # Compute train metrics
            if len(self.train_predictions) > 0 and len(self.train_labels) > 0:
                # Ensure lists are same length (safety check)
                min_len = min(len(self.train_predictions), len(self.train_labels))
                if min_len > 0:
                    train_metrics = compute_classification_metrics(
                        self.train_predictions[:min_len], 
                        self.train_labels[:min_len]
                    )
                    for key, value in train_metrics.items():
                        logs[f"train/{key}"] = value
                    # Clear stored predictions after logging
                    self.train_predictions.clear()
                    self.train_labels.clear()
            
            # Compute eval metrics
            if len(self.eval_predictions) > 0 and len(self.eval_labels) > 0:
                # Ensure lists are same length (safety check)
                min_len = min(len(self.eval_predictions), len(self.eval_labels))
                if min_len > 0:
                    eval_metrics = compute_classification_metrics(
                        self.eval_predictions[:min_len], 
                        self.eval_labels[:min_len]
                    )
                    for key, value in eval_metrics.items():
                        logs[f"eval/{key}"] = value
                    # Clear stored predictions after logging
                    self.eval_predictions.clear()
                    self.eval_labels.clear()
            
            self.step += 1
    
    # GRPOTrainer has different initialization parameters than GSPOTrainer
    if USE_GRPO:
        # GRPOTrainer uses processing_class and reward_funcs (plural)
        # Also requires dataset as list of dicts with "prompt" key
        trainer = GSPOTrainer(
            model=model,
            processing_class=tok,
            args=gspo_cfg,
            train_dataset=train_dataset_formatted,
            eval_dataset=eval_dataset_formatted,
            reward_funcs=[rf],  # GRPOTrainer expects a list of reward functions
        )
    else:
        trainer = GSPOTrainer(
            model=model,
            tokenizer=tok,
            args=gspo_cfg,
            train_dataset=train_dataset_formatted,
            eval_dataset=eval_dataset_formatted,
            reward_func=rf,
        )
    
    # Add metrics callback with references to the prediction lists
    metrics_callback = MetricsCallback(
        train_predictions, 
        train_labels_for_metrics, 
        eval_predictions, 
        eval_labels_for_metrics
    )
    trainer.add_callback(metrics_callback)
    
    return trainer
