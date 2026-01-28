"""
select_random.py - Random selection from instruction pool

Usage:
    python scripts/select_random.py --k 5000 --output data/selected/random_5k.json
    python scripts/select_random.py --k 20000 --output data/selected/random_20k.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_instruction_pool(pool_path: str) -> List[Dict]:
    """Load instruction pool from JSON file."""
    with open(pool_path, "r") as f:
        return json.load(f)


def random_select(data: List[Dict], k: int, seed: int = 42) -> List[Dict]:
    """Randomly select k samples from data."""
    random.seed(seed)
    
    if k >= len(data):
        print(f"Warning: k={k} >= pool size={len(data)}, returning all data")
        return data
    
    return random.sample(data, k)


def main():
    parser = argparse.ArgumentParser(description="Random subset selection")
    parser.add_argument("--pool", type=str, default="data/instruction_pool/openhermes.json", 
                        help="Path to instruction pool")
    parser.add_argument("--k", type=int, required=True, help="Number of samples to select")
    parser.add_argument("--output", type=str, required=True, help="Output path for selected data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    
    # Load instruction pool
    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = script_dir / pool_path
    
    print(f"Loading instruction pool from {pool_path}...")
    pool = load_instruction_pool(pool_path)
    print(f"Pool size: {len(pool)}")
    
    # Select random subset
    print(f"Selecting {args.k} random samples (seed={args.seed})...")
    selected = random_select(pool, args.k, args.seed)
    print(f"Selected: {len(selected)} samples")
    
    # Save selection
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
