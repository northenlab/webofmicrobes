# Qwen2.5 Fine-Tuning Guide: QLoRA, LoRA, and Continued Pretraining

This guide consolidates research-backed recommendations for fine-tuning Qwen2.5 models, moving beyond rules of thumb to specific, evidence-based settings.

---

## Table of Contents
1. [Quick Reference Table](#quick-reference-table)
2. [LoRA/QLoRA Settings](#loraqora-settings)
3. [Continued Pretraining](#continued-pretraining)
4. [Framework-Specific Configs](#framework-specific-configs)
5. [Common Pitfalls](#common-pitfalls)

---

## Quick Reference Table

| Parameter | LoRA | QLoRA | Continued Pretraining |
|-----------|------|-------|----------------------|
| Learning Rate | 2e-4 (start), 5e-6 (DPO/RLHF) | 2e-4 (start), 5e-6 (DPO/RLHF) | 1e-5 to 1e-4 with re-warming |
| Rank (r) | 8-64 (16 typical) | 8-64 (16 typical) | N/A |
| Alpha | r or 2×r | r or 2×r | N/A |
| Dropout | 0.05-0.1 | 0.05-0.1 | N/A |
| Epochs | 1-3 | 1-3 | Token-based (not epochs) |
| Batch Size (effective) | 16-32 | 16-32 | Large (128-2048) |
| Precision | bf16/fp16 | 4-bit NF4 | bf16 |

---

## LoRA/QLoRA Settings

### Target Modules for Qwen2.5

The official Qwen repository and research consensus recommend targeting **all linear layers** for best results:

```python
target_modules = [
    "q_proj",    # Query projection (attention)
    "k_proj",    # Key projection (attention)
    "v_proj",    # Value projection (attention)
    "o_proj",    # Output projection (attention)
    "gate_proj", # MLP gating
    "up_proj",   # MLP up projection
    "down_proj", # MLP down projection
]
```

**Research finding**: The QLoRA paper shows that applying LoRA to both attention AND MLP layers significantly outperforms attention-only configurations on RougeL scores.

**Minimal configuration** (if memory-constrained):
```python
target_modules = ["q_proj", "v_proj"]  # Attention only
```

### Rank and Alpha Settings

| Rank (r) | Use Case | Alpha Recommendation |
|----------|----------|---------------------|
| 4-8 | Simple tasks, small datasets | r or 2×r |
| 16 | General fine-tuning (recommended default) | 16-32 |
| 32-64 | Complex tasks, larger models | 32-128 |
| 64+ | Very complex/domain adaptation | 64-256 |

**Key insight from rsLoRA paper**: Use `use_rslora=True` for rank-stabilized LoRA, which scales alpha by sqrt(rank) for better training stability at higher ranks.

```python
# Recommended starting config
LoraConfig(
    r=16,
    lora_alpha=32,       # 2×rank
    lora_dropout=0.05,
    use_rslora=True,     # Enable rank-stabilized scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

### QLoRA Quantization Config

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Second quantization for memory savings
    bnb_4bit_quant_type="nf4",       # NormalFloat4 - best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
)
```

**LoRA vs QLoRA tradeoffs**:
- QLoRA: ~75% VRAM reduction, slightly slower, marginally less accurate
- LoRA (16-bit): Faster, slightly more accurate, ~4× more VRAM

### Learning Rate Guidelines

| Training Type | Learning Rate | Notes |
|--------------|---------------|-------|
| SFT (LoRA/QLoRA) | 2e-4 | Standard starting point |
| DPO/GRPO/RLHF | 5e-6 | Much lower for preference learning |
| Full fine-tuning | 1e-5 to 5e-5 | Lower than LoRA |

**Schedule**: Cosine decay with warmup is standard. Warmup ratio of 0.03-0.1 is typical.

### Training Duration

**Recommendation**: 1-3 epochs for instruction tuning datasets.

**Research finding**: Training beyond 3 epochs shows diminishing returns and increased overfitting risk. Monitor validation loss—stop when it starts increasing.

### Batch Size Configuration

```python
# Effective batch size = per_device_batch_size × gradient_accumulation_steps
per_device_train_batch_size = 4   # Limited by VRAM
gradient_accumulation_steps = 4   # Accumulate for stability
# Effective batch = 16

# For multi-GPU
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
# Effective batch = 8 × num_gpus
```

**Key insight**: Larger effective batch sizes (16-32) lead to smoother, more stable training.

---

## Continued Pretraining

Continued pretraining (CPT) adapts a pretrained model to new data/domains without full retraining.

### Learning Rate Schedule

The most effective approach is **re-warming and re-decaying**:

```
1. Re-warm: Linear warmup to peak LR (lower than original pretraining)
2. Stable: Hold at peak LR
3. Re-decay: Cosine decay to near-zero
```

**Recommended peak LR**: 10-50% of the original pretraining LR

```python
# Example for continued pretraining
learning_rate = 1e-5  # Much lower than SFT
lr_scheduler_type = "cosine"
warmup_ratio = 0.1    # 10% warmup
```

### Data Mixing Strategy

**Research-backed approach** from "Reuse, Don't Retrain" (2024):
1. Mix original pretraining data with new data (prevents catastrophic forgetting)
2. Typical ratio: 50-70% new data, 30-50% replay data
3. Phase transition: Start with more replay, shift to more target data

```python
# Data mixture example
data_mix = {
    "new_domain_data": 0.70,
    "original_pretrain_sample": 0.20,
    "instruction_data": 0.10,  # Maintains instruction-following
}
```

### Key Hyperparameters for CPT

| Parameter | Recommendation | Notes |
|-----------|----------------|-------|
| Learning Rate | 1e-5 to 5e-5 | Lower than original pretraining |
| Warmup | 5-10% of steps | Re-warm from low LR |
| Weight Decay | 0.1 | Standard AdamW setting |
| Precision | bf16 | Use bf16 if available |
| Optimizer | AdamW | β₁=0.9, β₂=0.95 |

### DeepSpeed Configuration

For CPT with Qwen2.5:
- **ZeRO-3**: Recommended for multi-GPU training (except Q-LoRA)
- **ZeRO-2**: Use for Q-LoRA (ZeRO-3 breaks gradient flow with LoRA adapters)

---

## Framework-Specific Configs

### Hugging Face PEFT + TRL

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
import torch

# QLoRA setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="./qwen25-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    max_seq_length=2048,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)
```

### LLaMA-Factory YAML Config

```yaml
# qwen25_lora_sft.yaml
### Model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
template: qwen

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all  # Targets all linear layers

### LoRA parameters
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
use_rslora: true

### Dataset
dataset: your_dataset
cutoff_len: 4096
preprocessing_num_workers: 16

### Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_train_epochs: 3.0
weight_decay: 0.1
max_grad_norm: 1.0

### Output
output_dir: saves/qwen25-7b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true

### Precision
bf16: true
flash_attn: auto

### Optional: Training on completions only (recommended)
train_on_prompt: false
```

### Unsloth (2x faster training)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_rslora=True,
    bias="none",
)
```

### NVIDIA NeMo

```python
from nemo.collections import llm

recipe = llm.qwen25_7b.finetune_recipe(
    name="qwen25_7b_finetuning",
    dir="/path/to/checkpoints",
    num_nodes=1,
    num_gpus_per_node=8,
    peft_scheme='lora',        # 'lora' or 'none' for full
    packed_sequence=True,      # Enable for throughput
)
```

---

## Common Pitfalls

### 1. BitsAndBytes Misconfiguration
```python
# WRONG - causes NaN losses
bnb_4bit_compute_dtype=torch.float16  # On Ampere+ GPUs

# CORRECT
bnb_4bit_compute_dtype=torch.bfloat16
```

### 2. ZeRO-3 with LoRA
```
RuntimeError: element 0 of tensors does not require grad
```
**Solution**: Use ZeRO-2 for LoRA/QLoRA training.

### 3. Overfitting Signs
- Validation loss increases while training loss decreases
- Model repeats training examples verbatim

**Solutions**:
- Reduce epochs (try 1-2)
- Increase dropout (0.1)
- Reduce learning rate
- Use weight averaging: merge original model with fine-tuned, divide by 2

### 4. Memory Issues

| Model Size | QLoRA VRAM | LoRA VRAM | Full FT VRAM |
|------------|------------|-----------|--------------|
| 0.5B | ~4GB | ~8GB | ~16GB |
| 3B | ~8GB | ~16GB | ~48GB |
| 7B | ~12GB | ~24GB | ~80GB |
| 14B | ~20GB | ~40GB | ~160GB |
| 72B | ~48GB | ~160GB | ~640GB |

### 5. Train on Completions Only
The QLoRA paper shows training only on assistant responses (not user prompts) increases accuracy by ~1%.

```python
# In SFTTrainer
dataset_text_field = "completion"  # Not full conversation
```

---

## Model-Specific Notes

### Qwen2.5 vs Qwen2.5-VL (Vision-Language)

For VL models, vision encoder modules also contain `gate_proj`, `up_proj`, `down_proj`. To target only the LLM:

```python
# Use regex to exclude vision encoder
target_modules = r"^(?!.*visual\.).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

# Or use explicit module names
target_modules = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]
```

---

## Summary: Recommended Starting Points

### For SFT with QLoRA (most common)
```
r=16, alpha=32, dropout=0.05
target_modules=all (7 modules)
lr=2e-4, epochs=2-3, batch=16
4-bit NF4, bf16 compute
```

### For DPO/RLHF
```
Same as above, but lr=5e-6
```

### For Continued Pretraining
```
Full precision (bf16)
lr=1e-5 with re-warming
Mix: 70% new data, 30% replay
Use ZeRO-3 (not with LoRA)
```

---

## References

- Unsloth Documentation: https://docs.unsloth.ai
- Qwen Official: https://qwen.readthedocs.io
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- "Simple and Scalable Strategies to Continually Pre-train LLMs" (2024)
- "Reuse, Don't Retrain" (2024)
- QLoRA Paper: Dettmers et al. (2023)
- rsLoRA Paper: Rank-Stabilized LoRA
