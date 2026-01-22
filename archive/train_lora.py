#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Advisor Personas
Takes JSONL training data and produces a LoRA adapter.
"""

import argparse
import time
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 512

# ============================================================================
# Training
# ============================================================================

def train_lora(
    data_path: str,
    output_dir: str = "./lora_output",
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Train a LoRA adapter on the given dataset"""
    
    print("="*60)
    print("LoRA Fine-tuning")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    print("\n[1/5] Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time() - t0:.1f}s")
    
    # Prepare for LoRA
    print("\n[2/5] Preparing LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\n[3/5] Loading training data...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  Loaded {len(dataset)} examples")
    
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
    print(f"  Tokenized {len(tokenized_dataset)} examples")
    
    # Training
    print("\n[4/5] Training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        learning_rate=learning_rate,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    
    # Save
    print("\n[5/5] Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Training time: {train_time/60:.1f} minutes")
    print(f"LoRA adapter saved to: {output_dir}")
    print(f"\nTo test inference:")
    print(f"  python3 test_lora.py --adapter {output_dir}")
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_dir

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter for advisor persona")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--output", type=str, default="./lora_output", help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    train_lora(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
