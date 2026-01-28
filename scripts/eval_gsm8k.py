"""
eval_gsm8k.py - Evaluate model on GSM8K benchmark

Usage:
    python scripts/eval_gsm8k.py --model_path Qwen/Qwen2.5-1.5B
    python scripts/eval_gsm8k.py --model_path checkpoints/random_5k
    python scripts/eval_gsm8k.py --model_path checkpoints/random_5k --num_samples 100
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    # Look for #### pattern first (GSM8K format)
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for "the answer is" pattern
    match = re.search(r"(?:the answer is|answer is|answer:)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for boxed answer (common in math)
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for last number in the response
    numbers = re.findall(r"([\d,]+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from GSM8K answer text."""
    # GSM8K format: reasoning #### answer
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def format_prompt(question: str) -> str:
    """Format question into model prompt."""
    prompt = f"""### Instruction:
Solve this math problem step by step. At the end, provide the final numerical answer after ####.

{question}

### Response:
"""
    return prompt


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 1,
    device: str = "cuda"
) -> Tuple[float, List[Dict]]:
    """Evaluate model on GSM8K dataset."""
    
    correct = 0
    total = 0
    results = []
    
    model.eval()
    
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        gold_answer = extract_gold_answer(item["answer"])
        
        # Format prompt
        prompt = format_prompt(question)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extract predicted answer
        pred_answer = extract_answer(response)
        
        # Check correctness
        is_correct = pred_answer is not None and pred_answer == gold_answer
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": pred_answer,
            "response": response,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def load_model_and_tokenizer(
    model_path: str,
    use_4bit: bool = False,
    device: str = "cuda"
):
    """Load model and tokenizer, handling both base models and LoRA checkpoints."""
    
    model_path = Path(model_path)
    
    # Check if this is a LoRA checkpoint (has adapter files)
    is_lora = (model_path / "adapter_config.json").exists() if model_path.exists() else False
    
    if is_lora:
        print(f"Loading LoRA checkpoint from {model_path}...")
        
        # Load adapter config to get base model
        with open(model_path / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B")
        
        print(f"Base model: {base_model_name}")
        
        # Setup quantization if needed
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        print(f"Loading base model from {model_path}...")
        
        # Setup quantization if needed
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to GSM8K test data (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output file for detailed results")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_4bit, device)
    
    # Load GSM8K data
    if args.data_path:
        print(f"Loading GSM8K from {args.data_path}...")
        with open(args.data_path, "r") as f:
            dataset = json.load(f)
    else:
        print("Loading GSM8K from HuggingFace...")
        hf_dataset = load_dataset("openai/gsm8k", "main", split="test")
        dataset = [{"question": item["question"], "answer": item["answer"]} for item in hf_dataset]
    
    # Limit samples if specified
    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate
    accuracy, results = evaluate_gsm8k(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device
    )
    
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {sum(1 for r in results if r['correct'])}/{len(results)}")
    print(f"{'='*50}")
    
    # Save detailed results if requested
    if args.output:
        script_dir = Path(__file__).parent.parent
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "model_path": args.model_path,
                "accuracy": accuracy,
                "num_samples": len(results),
                "results": results
            }, f, indent=2)
        print(f"Detailed results saved to {output_path}")
    
    return accuracy


if __name__ == "__main__":
    main()
