#!/usr/bin/env python3
"""
MAXIMUM TRAINING CONFIGURATION FOR M2 MACBOOK AIR (8GB)
Optimized for 6-8 hour training run with maximum quality
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("="*80)
    print(" MAXIMUM TRAINING: Qwen2.5-1.5B on M2 MacBook Air")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # MAXIMUM CONFIGURATION
    config = {
        # Model
        "model": "Qwen/Qwen2.5-1.5B-Instruct",

        # Data (using our local dataset with proper splits)
        "data": "./data",  # Has train.jsonl, valid.jsonl, test.jsonl

        # Batch & Memory
        "batch_size": 4,                    # Pushed up from 2
        "grad_accumulation_steps": 32,      # Effective batch = 128!
        "max_seq_length": 2048,             # 4x longer sequences

        # Training Duration (TARGET: 8 HOURS)
        "iters": 15000,                     # 25x more than before!
        "learning_rate": 0.0002,            # 2e-4 for longer training

        # Evaluation & Saving
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "save_every": 500,                  # More frequent checkpoints
        "val_batches": -1,

        # Output
        "adapter_path": "./adapters_maximum",

        # Optimization
        "seed": 42,
        "optimizer": "adamw",
    }

    print("\nConfiguration:")
    print("-"*80)
    print(f"  Model:                {config['model']}")
    print(f"  Dataset:              {config['data']}")
    print(f"  Batch size:           {config['batch_size']}")
    print(f"  Grad accumulation:    {config['grad_accumulation_steps']}")
    print(f"  Effective batch:      {config['batch_size'] * config['grad_accumulation_steps']}")
    print(f"  Max sequence length:  {config['max_seq_length']}")
    print(f"  Total iterations:     {config['iters']:,}")
    print(f"  Learning rate:        {config['learning_rate']}")
    print("-"*80)

    print("\nEstimates:")
    print(f"  Expected duration:    ~8 hours")
    print(f"  Peak memory:          ~5.0-5.5 GB")
    print(f"  Total tokens:         ~15 million")
    print(f"  Checkpoints saved:    Every 500 iterations")
    print("-"*80)

    # Build command for mlx_lm
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
        "--grad-checkpoint",  # Essential for memory!
        "--optimizer", config["optimizer"],
        "--num-layers", "-1",
        "--val-batches", str(config["val_batches"]),
    ]

    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    print(" TRAINING STARTED - THIS WILL TAKE ~8 HOURS")
    print("="*80)
    print("\nTips:")
    print("  - Keep your Mac plugged in")
    print("  - Disable sleep mode: System Settings > Lock Screen > Never")
    print("  - Monitor with: Activity Monitor > Memory")
    print("  - Loss will be reported every 10 iterations")
    print("  - Checkpoints saved every 500 iterations")
    print("\n" + "="*80 + "\n")

    # Run training
    try:
        subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"LoRA adapters saved to: {Path(config['adapter_path']).absolute()}")
        print("\nNext steps:")
        print("  1. Test the model: python test_model.py")
        print("  2. Compare checkpoints to see progression")
        print("  3. Merge adapters with base model if desired")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\nCommon issues:")
        print("  - Out of memory: Reduce batch_size or max_seq_length")
        print("  - Dataset error: Check internet connection")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Partial results saved to: {config['adapter_path']}")
        print("You can resume training by modifying the script to use --resume-adapter-file")
        sys.exit(1)


if __name__ == "__main__":
    main()
