#!/usr/bin/env python3
"""
Real LoRA Fine-tuning Script for Qwen2.5-1.5B
Uses mlx_lm for optimized training on M2 MacBook Air
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print(" Starting LoRA Fine-tuning of Qwen2.5-1.5B-Instruct")
    print("="*70)

    # Configuration
    config = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "data": "./data",
        "batch_size": 2,
        "iters": 600,
        "learning_rate": 0.0001,
        "grad_accumulation_steps": 8,
        "max_seq_length": 512,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 100,
        "adapter_path": "./adapters",
        "seed": 42,
    }

    print("\nConfiguration:")
    print("-"*70)
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print("-"*70)

    # Build command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", config["model"],
        "--train",
        "--data", config["data"],
        "--fine-tune-type", "lora",
        "--batch-size", str(config["batch_size"]),
        "--iters", str(config["iters"]),
        "--learning-rate", str(config["learning_rate"]),
        "--grad-accumulation-steps", str(config["grad_accumulation_steps"]),
        "--max-seq-length", str(config["max_seq_length"]),
        "--steps-per-report", str(config["steps_per_report"]),
        "--steps-per-eval", str(config["steps_per_eval"]),
        "--save-every", str(config["save_every"]),
        "--adapter-path", config["adapter_path"],
        "--seed", str(config["seed"]),
        "--grad-checkpoint",
        "--optimizer", "adamw",
        "--num-layers", "-1",
        "--val-batches", "-1",
    ]

    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    print(" Training started! This may take 30-90 minutes...")
    print("="*70 + "\n")

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print(" Training completed successfully!")
        print("="*70)
        print(f"\nLoRA adapters saved to: {Path(config['adapter_path']).absolute()}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
