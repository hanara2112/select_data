"""
train_lora.py - LoRA finetuning for instruction tuning

Usage:
    python scripts/train_lora.py --data data/selected/random_5k.json --output checkpoints/random_5k
    python scripts/train_lora.py --data data/selected/random_5k.json --output checkpoints/random_5k --use_4bit
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm
import yaml


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_path, "r") as f:
        return json.load(f)


def format_instruction(item: Dict) -> str:
    """Format instruction-response pair into training format."""
    # Handle both formats: instruction/response and question/answer
    if "instruction" in item:
        instruction = item["instruction"]
        response = item["response"]
    elif "question" in item:
        instruction = item["question"]
        response = item["answer"]
    else:
        raise ValueError(f"Unknown data format: {item.keys()}")
    
    # Simple chat format
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return text


def tokenize_data(
    data: List[Dict],
    tokenizer,
    max_length: int = 1024
) -> Dataset:
    """Tokenize training data."""
    
    def tokenize_fn(examples):
        texts = [format_instruction(item) for item in examples["data"]]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"data": data})
    
    # Tokenize in batches
    tokenized_dataset = dataset.map(
        lambda x: {
            "text": format_instruction(x["data"]),
        },
        remove_columns=["data"],
    )
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = tokenized_dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=True,
        batch_size=1000,
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA finetuning")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Training config")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent.parent
    
    # Load config
    config_path = script_dir / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {
            "model": {"name": "Qwen/Qwen2.5-1.5B", "dtype": "bfloat16"},
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, 
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            "training": {"batch_size": 4, "gradient_accumulation_steps": 4, 
                        "learning_rate": 2e-4, "num_epochs": 1, "max_seq_length": 1024}
        }
    
    # Override config with command line args
    model_name = args.model or config["model"]["name"]
    epochs = args.epochs or config["training"]["num_epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    
    print(f"Model: {model_name}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"4-bit quantization: {args.use_4bit}")
    
    # Load training data
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    
    print(f"\nLoading data from {data_path}...")
    data = load_training_data(data_path)
    print(f"Loaded {len(data)} training examples")
    
    # Setup quantization if needed
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize data
    print("\nTokenizing data...")
    max_seq_length = config["training"].get("max_seq_length", 1024)
    train_dataset = tokenize_data(data, tokenizer, max_length=max_seq_length)
    print(f"Tokenized dataset: {len(train_dataset)} examples")
    
    # Setup training arguments
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        warmup_ratio=config["training"].get("warmup_ratio", 0.03),
        weight_decay=config["training"].get("weight_decay", 0.01),
        logging_steps=config["training"].get("logging_steps", 10),
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print("Training complete!")


if __name__ == "__main__":
    main()
