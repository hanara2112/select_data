"""
quick_test.py - Quick sanity check that training and evaluation work

This script:
1. Creates a tiny subset (100 samples)
2. Trains for 1 epoch with small batch size
3. Evaluates on 20 GSM8K samples

Use this to verify the pipeline works before running full experiments.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --use_4bit  # For low VRAM GPUs
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Quick pipeline test")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--skip_download", action="store_true", help="Skip data download")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    data_dir = base_dir / "data"
    
    print("="*60)
    print("QUICK PIPELINE TEST")
    print("="*60)
    
    # Step 1: Download data (if needed)
    if not args.skip_download:
        if not (data_dir / "instruction_pool" / "openhermes.json").exists():
            print("\n[1/5] Downloading data...")
            subprocess.run([sys.executable, str(scripts_dir / "download_data.py")], check=True)
        else:
            print("\n[1/5] Data already exists, skipping download")
    else:
        print("\n[1/5] Skipping data download")
    
    # Step 2: Create tiny test subset
    print("\n[2/5] Creating tiny test subset (100 samples)...")
    pool_path = data_dir / "instruction_pool" / "openhermes.json"
    test_path = data_dir / "selected" / "test_100.json"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(pool_path, "r") as f:
        pool = json.load(f)
    
    random.seed(42)
    subset = random.sample(pool, min(100, len(pool)))
    
    with open(test_path, "w") as f:
        json.dump(subset, f)
    print(f"Created {test_path}")
    
    # Step 3: Test training
    print("\n[3/5] Testing training (1 epoch, tiny batch)...")
    checkpoint_path = base_dir / "checkpoints" / "test_100"
    
    train_cmd = [
        sys.executable,
        str(scripts_dir / "train_lora.py"),
        "--data", str(test_path),
        "--output", str(checkpoint_path),
        "--epochs", "1",
        "--batch_size", "1",
    ]
    if args.use_4bit:
        train_cmd.append("--use_4bit")
    
    subprocess.run(train_cmd, check=True)
    
    # Step 4: Test evaluation (baseline)
    print("\n[4/5] Testing baseline evaluation (20 samples)...")
    eval_cmd = [
        sys.executable,
        str(scripts_dir / "eval_gsm8k.py"),
        "--model_path", "Qwen/Qwen2.5-1.5B",
        "--num_samples", "20",
        "--output", str(base_dir / "results" / "test_baseline.json"),
    ]
    if args.use_4bit:
        eval_cmd.append("--use_4bit")
    
    subprocess.run(eval_cmd, check=True)
    
    # Step 5: Test evaluation (finetuned)
    print("\n[5/5] Testing finetuned evaluation (20 samples)...")
    eval_cmd = [
        sys.executable,
        str(scripts_dir / "eval_gsm8k.py"),
        "--model_path", str(checkpoint_path),
        "--num_samples", "20",
        "--output", str(base_dir / "results" / "test_finetuned.json"),
    ]
    
    subprocess.run(eval_cmd, check=True)
    
    # Print results
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    
    # Load and compare results
    with open(base_dir / "results" / "test_baseline.json", "r") as f:
        baseline = json.load(f)
    
    with open(base_dir / "results" / "test_finetuned.json", "r") as f:
        finetuned = json.load(f)
    
    print(f"\nBaseline accuracy (20 samples):  {baseline['accuracy']:.4f}")
    print(f"Finetuned accuracy (20 samples): {finetuned['accuracy']:.4f}")
    print("\nPipeline is working! Ready for full experiments.")
    print("\nNext steps:")
    print("  1. Run: python scripts/run_experiments.py --download")
    print("  2. Run: python scripts/run_experiments.py --baseline")
    print("  3. Run: python scripts/run_experiments.py --method random --k 5000")


if __name__ == "__main__":
    main()
