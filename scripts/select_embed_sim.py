"""
select_embed_sim.py - Embedding similarity-based selection

Selects instruction pool samples that are most similar to target task (GSM8K) examples.

Usage:
    python scripts/select_embed_sim.py --k 5000 --output data/selected/embed_5k.json
    python scripts/select_embed_sim.py --k 5000 --output data/selected/embed_5k.json --num_target 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_data(path: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_text_for_embedding(item: Dict) -> str:
    """Extract text from item for embedding."""
    # Handle both instruction/response and question/answer formats
    if "instruction" in item:
        return item["instruction"]
    elif "question" in item:
        return item["question"]
    else:
        raise ValueError(f"Unknown data format: {item.keys()}")


def compute_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings


def select_by_similarity(
    pool_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Select top-k samples most similar to target embedding."""
    # Compute cosine similarity (embeddings are normalized)
    similarities = np.dot(pool_embeddings, target_embedding)
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = similarities[top_k_indices]
    
    return top_k_indices, top_k_similarities


def main():
    parser = argparse.ArgumentParser(description="Embedding similarity selection")
    parser.add_argument("--pool", type=str, default="data/instruction_pool/openhermes.json",
                        help="Path to instruction pool")
    parser.add_argument("--target", type=str, default="data/gsm8k/train.json",
                        help="Path to target task data")
    parser.add_argument("--k", type=int, required=True, help="Number of samples to select")
    parser.add_argument("--output", type=str, required=True, help="Output path for selected data")
    parser.add_argument("--num_target", type=int, default=100,
                        help="Number of target examples to use for mean embedding")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    
    # Load instruction pool
    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = script_dir / pool_path
    
    print(f"Loading instruction pool from {pool_path}...")
    pool = load_data(pool_path)
    print(f"Pool size: {len(pool)}")
    
    # Load target data
    target_path = Path(args.target)
    if not target_path.is_absolute():
        target_path = script_dir / target_path
    
    print(f"Loading target data from {target_path}...")
    target_data = load_data(target_path)
    print(f"Target size: {len(target_data)}")
    
    # Limit target samples for computing mean embedding
    if args.num_target > 0 and args.num_target < len(target_data):
        import random
        random.seed(42)
        target_data = random.sample(target_data, args.num_target)
        print(f"Using {len(target_data)} target samples for mean embedding")
    
    # Load embedding model
    print(f"\nLoading embedding model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    
    # Extract texts
    print("\nExtracting texts...")
    pool_texts = [get_text_for_embedding(item) for item in pool]
    target_texts = [get_text_for_embedding(item) for item in target_data]
    
    # Compute target embeddings and mean
    print(f"\nComputing target embeddings ({len(target_texts)} examples)...")
    target_embeddings = compute_embeddings(target_texts, model, args.batch_size, show_progress=True)
    mean_target_embedding = np.mean(target_embeddings, axis=0)
    mean_target_embedding = mean_target_embedding / np.linalg.norm(mean_target_embedding)  # Normalize
    print(f"Mean target embedding shape: {mean_target_embedding.shape}")
    
    # Compute pool embeddings
    print(f"\nComputing pool embeddings ({len(pool_texts)} examples)...")
    pool_embeddings = compute_embeddings(pool_texts, model, args.batch_size, show_progress=True)
    print(f"Pool embeddings shape: {pool_embeddings.shape}")
    
    # Select top-k by similarity
    print(f"\nSelecting top-{args.k} by similarity...")
    top_k_indices, top_k_similarities = select_by_similarity(
        pool_embeddings, mean_target_embedding, args.k
    )
    
    # Get selected samples
    selected = [pool[i] for i in top_k_indices]
    
    # Add similarity scores to selected samples
    for i, idx in enumerate(top_k_indices):
        selected[i]["_similarity_score"] = float(top_k_similarities[i])
        selected[i]["_pool_index"] = int(idx)
    
    print(f"\nSelection statistics:")
    print(f"  Top similarity: {top_k_similarities[0]:.4f}")
    print(f"  Mean similarity: {np.mean(top_k_similarities):.4f}")
    print(f"  Min similarity: {top_k_similarities[-1]:.4f}")
    
    # Save selection
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved {len(selected)} samples to {output_path}")


if __name__ == "__main__":
    main()
