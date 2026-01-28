"""
download_data.py - Download GSM8K and OpenHermes-2.5 datasets

Usage:
    python scripts/download_data.py
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_gsm8k(output_dir: str):
    """Download GSM8K dataset."""
    print("Downloading GSM8K...")
    
    dataset = load_dataset("openai/gsm8k", "main")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train split
    train_data = []
    for item in tqdm(dataset["train"], desc="Processing train"):
        train_data.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved {len(train_data)} train examples to {output_path / 'train.json'}")
    
    # Save test split
    test_data = []
    for item in tqdm(dataset["test"], desc="Processing test"):
        test_data.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    with open(output_path / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved {len(test_data)} test examples to {output_path / 'test.json'}")


def download_openhermes(output_dir: str, max_samples: int = 100000):
    """Download OpenHermes-2.5 dataset."""
    print("Downloading OpenHermes-2.5...")
    
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process and save
    data = []
    for i, item in enumerate(tqdm(dataset, desc="Processing OpenHermes")):
        if i >= max_samples:
            break
        
        # Extract conversations
        conversations = item.get("conversations", [])
        if len(conversations) >= 2:
            # Find human/user and assistant messages
            user_msg = None
            assistant_msg = None
            
            for conv in conversations:
                role = conv.get("from", "").lower()
                if role in ["human", "user"] and user_msg is None:
                    user_msg = conv.get("value", "")
                elif role in ["gpt", "assistant"] and assistant_msg is None:
                    assistant_msg = conv.get("value", "")
            
            if user_msg and assistant_msg:
                data.append({
                    "instruction": user_msg,
                    "response": assistant_msg,
                    "source": item.get("source", "unknown")
                })
    
    with open(output_path / "openhermes.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} examples to {output_path / 'openhermes.json'}")


def main():
    base_dir = Path(__file__).parent.parent / "data"
    
    # Download GSM8K
    download_gsm8k(base_dir / "gsm8k")
    
    # Download OpenHermes (limit to 100k for initial experiments)
    download_openhermes(base_dir / "instruction_pool", max_samples=100000)
    
    print("\nData download complete!")
    print(f"GSM8K: {base_dir / 'gsm8k'}")
    print(f"Instruction Pool: {base_dir / 'instruction_pool'}")


if __name__ == "__main__":
    main()
