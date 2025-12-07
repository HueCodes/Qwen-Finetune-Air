# 🔥 MAXIMUM Training Configuration for M2 MacBook Air (8GB)

## Why These Specific Settings

### Model: Qwen2.5-1.5B-Instruct ✅
- **NOT 3B** because:
  ```
  3B FP16:     6.0 GB
  Activations: 2.5 GB (at seq=2048, batch=4, checkpointed)
  Total:       8.5 GB ❌ OOM RISK
  ```
- **1.5B is perfect:**
  ```
  1.5B FP16:   3.0 GB
  Activations: 1.8 GB (at seq=2048, batch=4, checkpointed)
  LoRA:        0.16 GB (rank=32 params + gradients + optimizer)
  Total:       5.0 GB ✅ SAFE with 3GB margin
  ```

### Batch Configuration
```python
batch_size = 4                 # Per-step batch
grad_accumulation = 32         # Steps before optimizer update
effective_batch = 128          # What the model actually sees
```

**Why this split:**
- Small `batch_size=4` keeps memory low per step
- Large `grad_accum=32` gives stable gradients (like training with batch=128)
- Best of both worlds!

**Why not bigger?**
- batch=8 would push activations to ~3.6GB, total ~6.8GB
- Still safe, but less margin for OS/background tasks
- Current settings are 99% optimal with safety buffer

### Sequence Length: 2048 tokens

**Why 2048:**
- Attention memory: O(n²) but MLX optimizes well
- At 2048: Most conversations fit in one context
- Longer = better understanding of context

**Why not 4096:**
```
Activations at 4096: ~3.5 GB
Total: ~6.7 GB
Risk: Some examples might be longer, causing spikes → OOM
```
- Could work but risky
- 2048 is sweet spot for M2 8GB

### LoRA Configuration
```python
rank = 32                      # 4x larger than typical
alpha = 64                     # 2 * rank (standard)
dropout = 0.05
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (essential)
    "gate_proj", "up_proj", "down_proj"       # MLP (bonus quality)
]
```

**Trainable parameters:**
- Rank 8:  ~2.5M params (0.16%)
- Rank 16: ~5.0M params (0.32%)
- Rank 32: ~10M params (0.65%) ← Our choice
- Rank 64: ~20M params (1.3%) - would work but slower

**Why rank=32:**
- Much more expressive than rank=8
- Still tiny compared to full fine-tuning
- Memory cost negligible (~150MB total)
- Training speed barely affected

### Training Duration: 15,000 iterations

**Time calculation:**
```
Seq length 2048 is 4x longer → ~4x slower per iteration
Expected speed: 0.4-0.6 it/sec (vs 1.7 it/sec at seq=512)

At 0.5 it/sec:
15,000 iters / 0.5 = 30,000 seconds = 8.3 hours ✅
```

**Why 15,000:**
- With effective batch=128, this is ~117,000 effective examples
- On guanaco-1k dataset, this is ~117 epochs (excellent convergence)
- On alpaca-52k dataset, this is ~2.25 epochs (good coverage)

### Dataset: mlabonne/guanaco-llama2-1k

**Why this dataset:**
- High-quality curated conversations
- 1,000 examples, clean and diverse
- Focused on instruction-following
- Good test for overfitting prevention

**Alternatives:**
```python
# More data, slightly lower quality
"yahma/alpaca-cleaned"          # 52k examples, general instructions

# Even higher quality, much larger
"OpenAssistant/oasst1"          # 161k messages, excellent but slower download
```

## Memory Breakdown (Detailed)

```
Component                           Size        Notes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model weights (FP16)                3.0 GB      Frozen during training
LoRA adapters (rank=32)             0.04 GB     Actual trainable params
LoRA gradients                      0.04 GB     For backpropagation
AdamW optimizer state (2 moments)   0.08 GB     Running averages
Activations (checkpointed)          1.8 GB      Saved for backward pass
Dataset in memory                   0.1 GB      Tokenized examples
Misc (MLX overhead, buffers)        0.1 GB      Framework overhead
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL PEAK                          5.2 GB      ✅ 2.8GB safety margin
```

## MLX-Specific Optimizations

**MLX is NOT PyTorch!** Different optimizations apply:

### 1. Unified Memory Architecture
```
Traditional GPU:     RAM ←→ VRAM (slow transfer)
M2 Unified Memory:   Single pool (instant access)
```
- MLX can use all 8GB efficiently
- No CPU→GPU transfer overhead
- Better memory utilization than CUDA on 8GB VRAM

### 2. MLX Lazy Evaluation
- Operations are lazy (not executed immediately)
- Graph optimization happens automatically
- Better memory reuse than eager PyTorch

### 3. No Flash Attention Needed
- MLX has optimized attention built-in
- Uses Metal Performance Shaders
- Already faster than vanilla PyTorch attention

### 4. Gradient Checkpointing
- `--grad-checkpoint` flag essential
- Trades computation for memory
- Saves every other layer's activations
- Recomputes during backward pass
- Reduces activation memory by ~50%

## What You CAN'T Do on MLX (vs CUDA)

❌ **4-bit/8-bit QLoRA during training** - MLX loads full FP16 model
   (Quantization works for inference, not training yet)

❌ **Flash Attention 2** - CUDA only, but MLX has equivalent speed

❌ **Unsloth** - CUDA only framework

❌ **DeepSpeed/FSDP** - Not applicable for single-device

✅ **What MLX DOES better:**
- Native M-series optimization
- Lower latency than CUDA on same memory
- Better power efficiency
- Simpler setup (no CUDA drivers)

## Pushing Even Harder (If You Want)

### Option A: Longer Sequences (Risky)
```python
max_seq_length = 3072  # vs 2048
# Adds ~1.2GB activations
# Total: ~6.4GB
# Risk: 10-20% chance of OOM on long examples
```

### Option B: Larger Batch (Safer)
```python
batch_size = 6         # vs 4
grad_accumulation = 21 # adjust to keep effective=126
# Adds ~0.9GB activations
# Total: ~6.1GB
# More stable, slightly faster
```

### Option C: Current Settings (RECOMMENDED)
```python
# What we have now
# 5.2GB peak, 2.8GB margin
# Rock solid, optimal for overnight run
```

## Training Expectations

### Loss Trajectory (Predicted)
```
Iter 0:      ~3.0    (Random)
Iter 1000:   ~1.5    (Learning patterns)
Iter 5000:   ~0.5    (Getting good)
Iter 10000:  ~0.2    (Converging)
Iter 15000:  ~0.1    (Well-trained)
```

### Validation Loss
- Expect val_loss higher than train_loss
- Gap of 0.3-0.8 is normal
- Gap > 2.0 indicates overfitting (unlikely with LoRA)

### What Success Looks Like
✅ Smooth decreasing loss
✅ Validation loss tracking training (with gap)
✅ No OOM errors
✅ Checkpoints saved regularly
✅ Memory stable around 5-5.5GB

### What Failure Looks Like
❌ Loss suddenly spikes
❌ Validation loss increases while training decreases
❌ Memory gradually climbing (leak)
❌ OOM crash

## Monitoring During Training

### Watch These Metrics
```bash
# Every 10 iterations you'll see:
Iter 100: Train loss 1.234, Learning Rate 2.000e-04, It/sec 0.523,
          Tokens/sec 1075.3, Trained Tokens 105,472, Peak mem 5.147 GB

Key metrics:
- Train loss: Should decrease smoothly
- It/sec: Speed (0.4-0.6 expected)
- Peak mem: Should stay 5.0-5.5GB
```

### Use Activity Monitor
```
Applications > Utilities > Activity Monitor > Memory tab
- Memory Pressure: Should stay GREEN
- Physical Memory: Should use ~6-7GB total (5.5GB for training + OS)
- Swap: Minimal (< 500MB)
```

If Memory Pressure goes YELLOW/RED:
- STOP training immediately
- Reduce batch_size to 2 or max_seq_length to 1024

## After Training

### Files Created
```
adapters_maximum/
├── adapters.safetensors           (Final, 40MB)
├── 0000500_adapters.safetensors  (Checkpoint 1)
├── 0001000_adapters.safetensors  (Checkpoint 2)
├── ...
└── adapter_config.json            (LoRA settings)
```

### Each Checkpoint
- Size: ~40MB (tiny!)
- Contains: Only LoRA A and B matrices
- Can be loaded onto base model instantly

### Testing the Model
```python
from mlx_lm import load, generate

# Load base model + your adapters
model, tokenizer = load(
    "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="./adapters_maximum"
)

# Generate
response = generate(model, tokenizer, prompt="Q: What is LoRA?\nA:", max_tokens=100)
```

## FAQ

**Q: Can I use 3B model?**
A: Technically possible with batch=1, seq=1024, but would be slower and riskier. 1.5B at these settings will give better results.

**Q: Why not use quantization?**
A: MLX doesn't support QLoRA-style training yet (as of Dec 2024). For inference yes, for training no.

**Q: Can I stop and resume?**
A: Yes! Add `--resume-adapter-file ./adapters_maximum/adapters.safetensors` to continue from last checkpoint.

**Q: What if I run out of memory?**
A: Reduce in this order:
   1. batch_size: 4→2
   2. max_seq_length: 2048→1024
   3. grad_accumulation: 32→16 (but keep effective batch high)

**Q: Can I train faster?**
A: Not without sacrificing quality:
   - Shorter sequences: faster but worse context
   - Fewer iterations: faster but undertrained
   - Current settings are optimal for 8-hour high-quality run

---

## Ready to Go?

Run this command:
```bash
uv run python train_maximum.py
```

Expected output:
- Initial model download: 5 minutes (if not cached)
- Training start: Immediate
- First iteration: ~2 seconds
- Sustained speed: ~0.5 it/sec
- Total time: ~8 hours
- Final model: Production-quality LoRA adapters

**DO NOT:**
- Close laptop lid (will sleep)
- Unplug power (will throttle)
- Run heavy apps during training

**DO:**
- Keep it plugged in
- Disable auto-sleep
- Let it run overnight
- Check progress occasionally

Good luck! 🚀
