"""
run_experiments.py - Run full experiment pipeline and generate results table

Usage:
    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --baseline
    python scripts/run_experiments.py --method random --k 5000
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.scripts_dir = base_dir / "scripts"
        self.data_dir = base_dir / "data"
        self.results_dir = base_dir / "results"
        self.checkpoints_dir = base_dir / "checkpoints"
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "selected").mkdir(parents=True, exist_ok=True)
        
        # Results file
        self.results_file = self.results_dir / "results.csv"
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return False
    
    def download_data(self) -> bool:
        """Download required datasets."""
        return self.run_command(
            [sys.executable, str(self.scripts_dir / "download_data.py")],
            "Downloading datasets"
        )
    
    def run_selection(self, method: str, k: int, output_name: str) -> Optional[Path]:
        """Run data selection."""
        output_path = self.data_dir / "selected" / f"{output_name}.json"
        
        if method == "random":
            cmd = [
                sys.executable,
                str(self.scripts_dir / "select_random.py"),
                "--k", str(k),
                "--output", str(output_path),
            ]
        elif method == "embed":
            cmd = [
                sys.executable,
                str(self.scripts_dir / "select_embed_sim.py"),
                "--k", str(k),
                "--output", str(output_path),
            ]
        else:
            print(f"Unknown selection method: {method}")
            return None
        
        success = self.run_command(cmd, f"Selecting {k} samples using {method}")
        return output_path if success else None
    
    def run_training(self, data_path: Path, checkpoint_name: str, use_4bit: bool = False) -> Optional[Path]:
        """Run LoRA training."""
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "train_lora.py"),
            "--data", str(data_path),
            "--output", str(checkpoint_path),
        ]
        
        if use_4bit:
            cmd.append("--use_4bit")
        
        success = self.run_command(cmd, f"Training on {data_path.name}")
        return checkpoint_path if success else None
    
    def run_evaluation(self, model_path: str, output_name: str, num_samples: int = -1) -> Optional[float]:
        """Run GSM8K evaluation."""
        output_path = self.results_dir / f"eval_{output_name}.json"
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "eval_gsm8k.py"),
            "--model_path", str(model_path),
            "--output", str(output_path),
        ]
        
        if num_samples > 0:
            cmd.extend(["--num_samples", str(num_samples)])
        
        success = self.run_command(cmd, f"Evaluating {model_path}")
        
        if success and output_path.exists():
            with open(output_path, "r") as f:
                result = json.load(f)
            return result["accuracy"]
        return None
    
    def save_result(self, method: str, k: int, accuracy: float):
        """Save result to CSV file."""
        file_exists = self.results_file.exists()
        
        with open(self.results_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "method", "num_samples", "gsm8k_accuracy"])
            writer.writerow([
                datetime.now().isoformat(),
                method,
                k,
                f"{accuracy:.4f}"
            ])
        
        print(f"\nResult saved: {method} | {k} samples | {accuracy:.4f} accuracy")
    
    def print_results_table(self):
        """Print current results table."""
        if not self.results_file.exists():
            print("\nNo results yet.")
            return
        
        print(f"\n{'='*50}")
        print("RESULTS TABLE")
        print(f"{'='*50}")
        print(f"{'Method':<15} {'#Samples':<10} {'GSM8K Acc':<12}")
        print(f"{'-'*50}")
        
        with open(self.results_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(f"{row['method']:<15} {row['num_samples']:<10} {row['gsm8k_accuracy']:<12}")
        
        print(f"{'='*50}")
    
    def run_baseline(self, model_name: str = "Qwen/Qwen2.5-1.5B", num_samples: int = -1) -> Optional[float]:
        """Run baseline evaluation (no finetuning)."""
        accuracy = self.run_evaluation(model_name, "baseline", num_samples)
        if accuracy is not None:
            self.save_result("Base", 0, accuracy)
        return accuracy
    
    def run_full_experiment(self, method: str, k: int, use_4bit: bool = False, eval_samples: int = -1):
        """Run full experiment: selection -> training -> evaluation."""
        output_name = f"{method}_{k}"
        
        # Selection
        data_path = self.run_selection(method, k, output_name)
        if data_path is None:
            print(f"Selection failed for {method}_{k}")
            return None
        
        # Training
        checkpoint_path = self.run_training(data_path, output_name, use_4bit)
        if checkpoint_path is None:
            print(f"Training failed for {method}_{k}")
            return None
        
        # Evaluation
        accuracy = self.run_evaluation(str(checkpoint_path), output_name, eval_samples)
        if accuracy is not None:
            method_name = "Random" if method == "random" else "EmbedSim"
            self.save_result(method_name, k, accuracy)
        
        return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no finetuning)")
    parser.add_argument("--method", type=str, choices=["random", "embed"], help="Selection method")
    parser.add_argument("--k", type=int, help="Number of samples to select")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--eval_samples", type=int, default=-1, help="Number of eval samples (-1 = all)")
    parser.add_argument("--results", action="store_true", help="Show results table")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    runner = ExperimentRunner(base_dir)
    
    if args.results:
        runner.print_results_table()
        return
    
    if args.download:
        runner.download_data()
    
    if args.baseline:
        runner.run_baseline(num_samples=args.eval_samples)
    
    if args.method and args.k:
        runner.run_full_experiment(args.method, args.k, args.use_4bit, args.eval_samples)
    
    if args.all:
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60)
        
        # Download data
        runner.download_data()
        
        # Baseline
        runner.run_baseline(num_samples=args.eval_samples)
        
        # Random selection experiments
        for k in [5000, 20000]:
            runner.run_full_experiment("random", k, args.use_4bit, args.eval_samples)
        
        # Embedding similarity experiments
        runner.run_full_experiment("embed", 5000, args.use_4bit, args.eval_samples)
        
        # Print final results
        runner.print_results_table()
    
    # Always show current results at the end
    if not args.results:
        runner.print_results_table()


if __name__ == "__main__":
    main()
