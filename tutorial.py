#!/usr/bin/env python3
"""
Interactive Tutorial: Understanding LoRA Fine-tuning Step-by-Step
This script teaches you the core concepts with hands-on examples
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def subsection(title):
    """Print a subsection header"""
    print(f"\n>>> {title}")
    print("-" * 70)


def tutorial_1_matrices():
    """Tutorial 1: Understanding Matrix Multiplication"""
    section("Tutorial 1: Matrix Multiplication Basics")

    print("""
In neural networks, everything is matrix multiplication!
Let's see how a simple linear layer works.
    """)

    subsection("Example: Text → Hidden Representation")

    # Simulate a simple embedding
    input_dim = 4    # 4-dimensional input (simplified token embedding)
    hidden_dim = 6   # 6-dimensional hidden layer

    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")

    # Create a weight matrix
    W = mx.random.normal((input_dim, hidden_dim))
    print(f"\nWeight matrix W shape: {W.shape}")
    print(f"Number of parameters: {input_dim * hidden_dim} = {input_dim * hidden_dim}")

    # Create an input vector
    x = mx.array([1.0, 0.5, -0.3, 0.8])
    print(f"\nInput vector x: {x}")
    print(f"Input shape: {x.shape}")

    # Matrix multiplication
    output = x @ W  # @ is matrix multiplication in Python
    print(f"\nOutput = x @ W")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")

    print("""
💡 Key Insight:
   - The weight matrix W transforms our input
   - In LLMs, these matrices are HUGE (e.g., 1536 × 1536)
   - Fine-tuning normally updates ALL these parameters
    """)


def tutorial_2_lora():
    """Tutorial 2: Understanding LoRA"""
    section("Tutorial 2: How LoRA Works")

    print("""
LoRA doesn't change the original weights. Instead, it adds a small
correction through low-rank matrices A and B.
    """)

    subsection("Traditional Approach vs LoRA")

    hidden_dim = 1536  # Typical for Qwen2.5-1.5B

    print(f"Hidden dimension: {hidden_dim}")

    # Traditional approach
    W = mx.random.normal((hidden_dim, hidden_dim))
    traditional_params = hidden_dim * hidden_dim
    print(f"\n1. Traditional Fine-tuning:")
    print(f"   Weight matrix W: {hidden_dim} × {hidden_dim}")
    print(f"   Parameters to train: {traditional_params:,}")
    print(f"   Memory for gradients: {traditional_params * 4 / 1e6:.2f} MB (FP32)")

    # LoRA approach
    rank = 8  # LoRA rank
    print(f"\n2. LoRA Approach (rank={rank}):")
    print(f"   Original W: {hidden_dim} × {hidden_dim} (FROZEN ❄️)")

    A = mx.random.normal((hidden_dim, rank))
    B = mx.zeros((rank, hidden_dim))

    lora_params = (hidden_dim * rank) + (rank * hidden_dim)
    print(f"   Matrix A: {hidden_dim} × {rank}")
    print(f"   Matrix B: {rank} × {hidden_dim}")
    print(f"   Parameters to train: {lora_params:,}")
    print(f"   Memory for gradients: {lora_params * 4 / 1e6:.2f} MB (FP32)")

    reduction = traditional_params / lora_params
    print(f"\n   💰 Reduction: {reduction:.1f}x fewer parameters!")
    print(f"   📊 Trainable ratio: {(lora_params/traditional_params)*100:.2f}%")

    subsection("How LoRA Modifies the Output")

    # Create a sample input
    x = mx.random.normal((1, hidden_dim))

    # Original output
    original_output = x @ W
    print(f"\nOriginal output: x @ W")
    print(f"Shape: {original_output.shape}")

    # LoRA correction
    lora_correction = (x @ A) @ B
    print(f"\nLoRA correction: (x @ A) @ B")
    print(f"Shape: {lora_correction.shape}")

    # Final output
    alpha = 16  # scaling factor
    scaling = alpha / rank
    final_output = original_output + (lora_correction * scaling)

    print(f"\nFinal output: original + (lora_correction * {scaling})")
    print(f"Shape: {final_output.shape}")

    print("""
💡 Key Insights:
   - W stays frozen (no gradients computed)
   - Only A and B are updated during training
   - The rank 'r' controls the capacity and cost
   - Smaller rank = faster, less memory, but less expressive
   - Typical ranks: 4-64 (8 is common)
    """)


def tutorial_3_training_loop():
    """Tutorial 3: The Training Loop"""
    section("Tutorial 3: Training Loop Explained")

    print("""
This is what happens during each training step.
We'll walk through it with a tiny example.
    """)

    subsection("Setup")

    vocab_size = 100    # Simplified vocabulary
    hidden_dim = 16     # Simplified hidden dimension
    rank = 4            # LoRA rank

    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"LoRA rank: {rank}")

    # LoRA matrices (what we're training)
    A = mx.random.normal((hidden_dim, rank)) * 0.01
    B = mx.zeros((rank, hidden_dim))

    subsection("Step 1: Forward Pass")

    print("""
Input: Token IDs [5, 12, 23, 42]
Goal: Predict next token
    """)

    # Simplified - in reality this goes through embedding, attention, etc.
    x = mx.random.normal((1, hidden_dim))  # Pretend this came from earlier layers
    print(f"Hidden representation: shape {x.shape}")

    # Apply LoRA
    output = (x @ A) @ B
    print(f"After LoRA: shape {output.shape}")

    # Project to vocabulary (simplified)
    logits = output @ mx.random.normal((hidden_dim, vocab_size))
    print(f"Logits (scores for each token): shape {logits.shape}")

    subsection("Step 2: Compute Loss")

    print("""
We compare the model's prediction with the correct answer.
Loss function: Cross-entropy (measures how wrong we are)
    """)

    target = 42  # The correct next token
    print(f"Target token: {target}")
    print(f"Model's prediction: argmax(logits) = {mx.argmax(logits[0])}")

    # Simplified loss
    print("\nLoss = -log(probability of correct token)")
    print("Lower loss = better predictions")

    subsection("Step 3: Backward Pass (Gradients)")

    print("""
Gradients tell us how to adjust A and B to reduce the loss.

Gradient = ∂Loss/∂Parameter

This is computed automatically by the framework (MLX, PyTorch, etc.)
using the chain rule from calculus.
    """)

    print("\nFor our LoRA matrices:")
    print("  gradient_A = ∂Loss/∂A")
    print("  gradient_B = ∂Loss/∂B")

    subsection("Step 4: Update Parameters")

    learning_rate = 0.001
    print(f"\nLearning rate: {learning_rate}")

    print("\nUpdate rule:")
    print("  A_new = A_old - learning_rate * gradient_A")
    print("  B_new = B_old - learning_rate * gradient_B")

    # Simulate gradient (random for demo)
    gradient_A = mx.random.normal(A.shape) * 0.01
    gradient_B = mx.random.normal(B.shape) * 0.01

    A_updated = A - learning_rate * gradient_A
    B_updated = B - learning_rate * gradient_B

    print(f"\nParameters updated!")
    print(f"  A changed by: {mx.mean(mx.abs(A - A_updated)):.6f}")
    print(f"  B changed by: {mx.mean(mx.abs(B - B_updated)):.6f}")

    print("""
💡 This repeats for every batch, every epoch!
   - Batch: A group of training examples processed together
   - Epoch: One complete pass through the entire dataset
    """)


def tutorial_4_hyperparameters():
    """Tutorial 4: Hyperparameters Explained"""
    section("Tutorial 4: Hyperparameters Deep Dive")

    subsection("1. LoRA Rank (r)")

    print("""
The rank determines the "capacity" of the LoRA adaptation.

Low rank (r=4):
  ✓ Faster training
  ✓ Less memory
  ✓ Smaller adapter files
  ✗ Less expressive
  → Good for: Simple tasks, limited data

High rank (r=32):
  ✓ More expressive
  ✓ Better for complex tasks
  ✗ Slower training
  ✗ More memory
  → Good for: Complex tasks, lots of data

Sweet spot: r=8 to r=16 for most tasks
    """)

    subsection("2. Learning Rate")

    print("""
Controls how big of a step we take during optimization.

Too high (e.g., 0.01):
  - Training unstable
  - Loss might increase
  - Model might not converge

Too low (e.g., 1e-6):
  - Training very slow
  - Might get stuck
  - Takes forever to learn

Good range for LoRA: 1e-4 to 3e-4

Example schedule:
  - Start: 3e-4
  - Middle: 1e-4 (after 50% of training)
  - End: 3e-5 (fine-tuning)
    """)

    subsection("3. Batch Size")

    print("""
Number of examples processed before updating parameters.

Small batch (1-2):
  ✓ Less memory
  ✓ More frequent updates
  ✗ Noisy gradients
  ✗ Slower (less parallelism)

Large batch (16-32):
  ✓ Smoother gradients
  ✓ Better parallelism
  ✗ More memory
  ✗ Might overfit

For M2 MacBook Air: 2-8 is typical
Memory limited! Start small and increase.
    """)

    subsection("4. Number of Epochs")

    print("""
How many times to go through the entire dataset.

Too few (1-2):
  ✗ Underfit - model hasn't learned enough

Too many (20+):
  ✗ Overfit - model memorizes training data
  ✗ Poor generalization

Signs of overfitting:
  - Training loss keeps decreasing
  - Validation loss starts increasing
  → Stop training!

Good practice: 3-10 epochs, monitor validation loss
    """)


def tutorial_5_memory():
    """Tutorial 5: Memory Management"""
    section("Tutorial 5: Memory Management on M2 MacBook Air")

    print("""
Your M2 MacBook Air has unified memory shared between CPU and GPU.
Let's calculate what fits in memory.
    """)

    subsection("Memory Breakdown")

    # Model size
    model_params = 1.5e9  # 1.5B parameters
    bytes_per_param_fp32 = 4
    bytes_per_param_fp16 = 2

    model_size_fp32 = model_params * bytes_per_param_fp32 / 1e9
    model_size_fp16 = model_params * bytes_per_param_fp16 / 1e9

    print(f"Qwen2.5-1.5B model:")
    print(f"  FP32 (full precision): {model_size_fp32:.2f} GB")
    print(f"  FP16 (half precision): {model_size_fp16:.2f} GB")

    # LoRA adapters
    hidden_size = 1536
    num_layers = 28
    rank = 8
    num_target_modules = 4  # q, k, v, o projections

    lora_params = num_layers * num_target_modules * 2 * hidden_size * rank
    lora_size = lora_params * bytes_per_param_fp32 / 1e9

    print(f"\nLoRA adapters (rank={rank}):")
    print(f"  Parameters: {lora_params:,}")
    print(f"  Size: {lora_size:.2f} GB")

    # Gradients
    gradient_size = lora_size  # Same size as parameters

    print(f"\nGradients (for LoRA params):")
    print(f"  Size: {gradient_size:.2f} GB")

    # Optimizer state (Adam keeps running averages)
    optimizer_state = lora_size * 2  # Adam stores 2 moments

    print(f"\nOptimizer state (Adam):")
    print(f"  Size: {optimizer_state:.2f} GB")

    # Activations (depends on batch size and sequence length)
    batch_size = 4
    seq_length = 512
    activation_size = (batch_size * seq_length * hidden_size * num_layers *
                      bytes_per_param_fp16 / 1e9)

    print(f"\nActivations (batch={batch_size}, seq={seq_length}):")
    print(f"  Size: {activation_size:.2f} GB")

    # Total
    total = model_size_fp16 + lora_size + gradient_size + optimizer_state + activation_size

    print(f"\n{'─'*70}")
    print(f"TOTAL MEMORY USAGE: {total:.2f} GB")
    print(f"{'─'*70}")

    available_memory = 8
    print(f"\nAvailable on M2 MacBook Air: ~{available_memory} GB")
    print(f"System overhead: ~1-2 GB")
    print(f"Available for ML: ~{available_memory - 2} GB")

    if total > (available_memory - 2):
        print("\n⚠️  Might be tight! Strategies:")
        print("  1. Reduce batch size to 2 or 1")
        print("  2. Reduce sequence length to 256")
        print("  3. Use gradient checkpointing")
        print("  4. Use lower precision (FP16/BF16)")
    else:
        print("\n✓ Should fit! You're good to go.")

    subsection("Memory Optimization Tips")

    print("""
1. Batch Size:
   Start with 1, gradually increase

2. Sequence Length:
   Shorter sequences = less memory
   Most tasks work fine with 256-512 tokens

3. Gradient Accumulation:
   Simulate larger batches without memory cost
   Example: batch=1, accumulate 4 steps = effective batch of 4

4. Mixed Precision:
   Train in FP16 instead of FP32
   2x memory savings!

5. Gradient Checkpointing:
   Trade computation for memory
   Recompute activations instead of storing them
    """)


def main():
    """Run all tutorials"""
    print("="*70)
    print(" Welcome to the LoRA Fine-tuning Tutorial! ")
    print("="*70)
    print("""
This interactive tutorial will teach you:
  1. Matrix multiplication in neural networks
  2. How LoRA works under the hood
  3. The training loop step-by-step
  4. Hyperparameters and their effects
  5. Memory management on M2 Mac

Each section includes hands-on examples with code.
Let's get started!
    """)

    input("\nPress Enter to start Tutorial 1...")
    tutorial_1_matrices()

    input("\n\nPress Enter to continue to Tutorial 2...")
    tutorial_2_lora()

    input("\n\nPress Enter to continue to Tutorial 3...")
    tutorial_3_training_loop()

    input("\n\nPress Enter to continue to Tutorial 4...")
    tutorial_4_hyperparameters()

    input("\n\nPress Enter to continue to Tutorial 5...")
    tutorial_5_memory()

    section("Conclusion")
    print("""
Congratulations! You now understand:
  ✓ Why LoRA is efficient (low-rank matrices)
  ✓ How training works (forward, loss, backward, update)
  ✓ How to choose hyperparameters
  ✓ Memory constraints and optimization

Next steps:
  1. Run: uv run python train_simple.py
  2. Read: README.md for more details
  3. Modify: train.py to implement full training
  4. Experiment: Try different datasets and hyperparameters

Happy learning! 🚀
    """)


if __name__ == "__main__":
    main()
