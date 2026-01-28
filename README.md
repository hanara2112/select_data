# Data Selection Pipeline for Instruction Tuning

A minimal, working pipeline for data selection research in instruction tuning.

## Project Structure

```
data-selection-pipeline/
├── data/
│   ├── gsm8k/           # Target task data
│   ├── instruction_pool/ # Source instruction data (OpenHermes-2.5)
│   └── selected/         # Selected subsets
├── scripts/
│   ├── train_lora.py     # LoRA finetuning
│   ├── eval_gsm8k.py     # GSM8K evaluation
│   ├── select_random.py  # Random selection
│   └── select_embed_sim.py # Embedding similarity selection
├── configs/
│   └── train_config.yaml # Training configuration
└── results/
    └── results.csv       # Experiment results
```

## Quick Start

### 1. Setup Environment
```bash
pip install torch transformers datasets peft accelerate bitsandbytes sentence-transformers tqdm
```

### 2. Download Data
```bash
python scripts/download_data.py
```

### 3. Run Experiments
```bash
# Baseline (no finetuning)
python scripts/eval_gsm8k.py --model_path Qwen/Qwen2.5-1.5B

# Random selection
python scripts/select_random.py --k 5000 --output data/selected/random_5k.json
python scripts/train_lora.py --data data/selected/random_5k.json --output checkpoints/random_5k
python scripts/eval_gsm8k.py --model_path checkpoints/random_5k

# Embedding similarity selection
python scripts/select_embed_sim.py --k 5000 --output data/selected/embed_5k.json
python scripts/train_lora.py --data data/selected/embed_5k.json --output checkpoints/embed_5k
python scripts/eval_gsm8k.py --model_path checkpoints/embed_5k
```

## Fixed Choices

- **Model**: Qwen2.5-1.5B
- **Target Task**: GSM8K
- **Instruction Pool**: OpenHermes-2.5
- **Selection Sizes**: k = 5,000 and k = 20,000
- **Training**: LoRA (r=16, alpha=32)
