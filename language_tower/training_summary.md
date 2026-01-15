# SoilFM Language Tower Training Summary
**Continued Pretraining of Qwen2.5-14B-Instruct on Soil Science Literature**

Date: December 2025
Model: Qwen2.5-14B-Instruct
Method: QLoRA (4-bit quantization, LoRA fine-tuning)
Platform: A100 PCIe 80GB

---

## 1. Training Corpus Construction

### 1.1 Data Sources

The training corpus was assembled from four primary sources of soil science domain knowledge:

| Source | Type | Proportion | Train Examples | Val Examples |
|--------|------|-----------|----------------|--------------|
| **PubMed Central (PMC)** | Scientific literature | 55% | 110,000 | 5,500 |
| **Wikipedia** | General knowledge | 20% | 40,000 | 2,000 |
| **USDA Soil Survey Manual** | Technical manual | 10% | 20,000 | 1,000 |
| **Wikipedia General Biology** | General biology | 15% | 30,000 | 1,500 |
| **TOTAL** | | **100%** | **200,000** | **10,000** |

### 1.2 Corpus Preparation

**Original Full Corpus:**
- Training: 666,210 examples
- Validation: 35,064 examples
- Format: JSONL with fields: `text`, `source`, `metadata`

**Preprocessing Steps:**

1. **Soil Survey Manual Cleaning:**
   - Removed page headers, footers, and index content
   - Filtered 57 of 435 chunks (13%)
   - Result: 378 clean chunks with 14.3% token reduction

2. **Source Field Standardization:**
   - Mapped source labels to canonical names:
     - `'pmc'` → PMC
     - `'wikipedia'` → Wikipedia
     - `'soil_survey_manual'` → Soil Survey Manual
     - `'wikipedia_general_bio'` → Wikipedia General Biology

### 1.3 Stratified Subsampling to 200K

**Rationale:**
- Full 666K corpus would require extended training time for 1 epoch
- Subsampled to 200K for faster iteration
- Preserved exact source distribution ratios

**Method:**
- **Script:** `subsample_dataset.py`
- **Random seed:** 42 (for reproducibility)
- **Strategy:** Stratified random sampling within each source category

```python
target_ratios = {
    'pmc': 0.55,
    'wiki': 0.20,
    'manual': 0.10,
    'gen_bio': 0.15,
}

# For each source, sample proportionally
for source, ratio in target_ratios.items():
    n_samples = int(target_size * ratio)
    samples = random.sample(data_by_source[source], n_samples)
```

**Output Files:**
- Training: `soilfm_train_200k.jsonl` (200,000 examples)
- Validation: `soilfm_validation_10k.jsonl` (10,000 examples)

---

## 2. Training Configuration

### 2.1 Model Architecture

**Base Model:** Qwen/Qwen2.5-14B-Instruct
- Parameters: 14.2 billion
- Architecture: Transformer decoder (GPT-style)
- Vocabulary: 151,936 tokens
- Context length: 32,768 tokens

**Quantization:** 4-bit NF4 (BitsAndBytes)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

**LoRA Configuration:**
```python
LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.05,
    use_rslora=True,        # Rank-stabilized LoRA
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
        'gate_proj', 'up_proj', 'down_proj'       # MLP
    ],
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
```

**Trainable Parameters:** ~263M (1.85% of total model)

### 2.2 Training Hyperparameters

**Optimization:**
- Optimizer: PagedAdamW8bit (memory-efficient)
- Learning rate: 2e-5
- LR schedule: Cosine with 10% warmup
- Batch size: 2 per GPU
- Gradient accumulation: 64 steps
- **Effective batch size: 128** (2 × 64)
- Max gradient norm: 1.0 (gradient clipping)
- Weight decay: 0.01
- Precision: BF16 mixed precision

**Training Duration:**
- Epochs: 1
- Total steps: 1,562 (200K examples ÷ 128 effective batch)
- Actual steps completed: 1,500
- Evaluation frequency: Every 500 steps
- Checkpoint frequency: Every 500 steps

**Key Optimizations:**
- **Gradient checkpointing: DISABLED** (speed over memory)
- **Flash Attention 2: ENABLED** (2-3x attention speedup)
- Result: 1.54 sec/iteration (vs 7.38 sec with gradient checkpointing)

### 2.3 Training Infrastructure

**Hardware:**
- GPU: NVIDIA A100 PCIe 80GB
- RAM: 125GB
- Disk: 300GB

**Software Stack:**
- PyTorch: 2.x with CUDA
- Transformers: 4.46+
- PEFT: Latest (for LoRA)
- BitsAndBytes: 4-bit quantization
- Flash Attention 2: Optimized attention

**Training Script:**
- **Location:** `train_soilfm_cpt_MANUAL.py`
- **Type:** Manual PyTorch training loop (not HuggingFace Trainer)
- **Rationale:** HuggingFace Trainer had compatibility issues; manual loop provided better control

---

## 3. Training Results

### 3.1 Loss Curves

**Validation Loss** (primary metric):
- Step 500:  **1.7369**
- Step 1000: **1.6281**
- Step 1500: **1.6130** ← Best checkpoint
- **Total improvement: 7.2%** (1.737 → 1.613)

**Training Loss:**
- Displayed: ~120 → ~100
- Note: Due to logging bug (multiplied by gradient_accumulation factor), actual training loss was ~94→78
- Validation loss is the accurate metric

**Key Observations:**
1. Validation loss decreased steadily throughout training
2. No overfitting observed (loss still decreasing at end)
3. No gradient explosions or NaN losses
4. Stable gradient norms (0.2-0.8 range)

### 3.2 Training Stability

**Gradient Norms:**
- Range: 0.2 - 0.8
- No spikes above 1.0 (max_grad_norm threshold)
- Indicates healthy gradient flow

**Learning Rate Schedule:**
- Warmed up to 2e-5 over first 10% of training
- Smoothly decayed via cosine schedule

**Memory Usage:**
- Peak VRAM: 54GB / 80GB (67% utilization)
- Stable throughout training
- No OOM errors after reducing batch size from 8 to 2

### 3.3 Training Performance

**Duration:**
- Total training time: ~42 hours
- Speed: 1.54 seconds/iteration
- Total iterations: ~100,000

**Performance Improvements:**
- Initial speed with gradient checkpointing: 7.38 sec/it
- After optimizations: 1.54 sec/it
- 5x speedup achieved

---

## 4. Model Outputs & Artifacts

### 4.1 HuggingFace Model Repositories

**LoRA Adapter (Recommended for Research):**
- **Repository:** https://huggingface.co/northenlab/soilfm-qwen2.5-14b-qlora
- **Size:** 263MB
- **Format:** PEFT LoRA adapter
- **Usage:**
  ```python
  from transformers import AutoModelForCausalLM
  from peft import PeftModel

  base = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-14B-Instruct",
      load_in_4bit=True,
      device_map="auto"
  )
  model = PeftModel.from_pretrained(base, "northenlab/soilfm-qwen2.5-14b-qlora")
  ```

**Merged Full Model (Recommended for Production):**
- **Repository:** https://huggingface.co/northenlab/soilfm-qwen2.5-14b-literature-cpt
- **Size:** 28GB
- **Format:** Standalone model (LoRA merged into base)
- **Usage:**
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model = AutoModelForCausalLM.from_pretrained(
      "northenlab/soilfm-qwen2.5-14b-literature-cpt",
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained(
      "northenlab/soilfm-qwen2.5-14b-literature-cpt"
  )
  ```

### 4.2 Checkpoints Saved

**Best Checkpoint:** Step 1500, validation loss = 1.6130

**Files:**
- `adapter_model.safetensors` (263MB) - LoRA weights
- `adapter_config.json` - LoRA configuration
- Tokenizer files (vocab.json, merges.txt, etc.)
- README.md

---

## 5. Qualitative Evaluation

### 5.1 Generation Test

**Test Prompt:**
```
The role of root exudates in shaping rhizosphere microbial communities involves
```

**Model Output:**
```
The role of root exudates in shaping rhizosphere microbial communities involves
the emission of carbon compounds that are not directly involved in plant growth
and development. These compounds can be classified into three categories: primary
metabolites, secondary metabolites, and phytohormones. Primary metabolites are
essential for plant growth and development, while secondary metabolites are not
directly involved in these processes but may have defensive or signaling functions.
Phytohormones are chemical messengers that regulate plant growth and development.

In addition to influencing the composition of the rhizosphere microbiome,
```

**Assessment:**
- Domain-appropriate terminology (root exudates, rhizosphere, primary/secondary metabolites, phytohormones)
- Coherent scientific explanation
- Appropriate technical depth
- Evidence of successful domain adaptation

---

## 6. Technical Issues & Resolutions

### 6.1 Training Speed Issues

**Problem:** Initial training speed was 7.38 sec/iteration (185 hours for full dataset)

**Solution:**
1. Disabled gradient checkpointing: `prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)`
2. Enabled Flash Attention 2: `attn_implementation="flash_attention_2"` in model loading

**Result:** 5x speedup (7.38s → 1.54s per iteration)

### 6.2 Out of Memory (OOM) Errors

**Problem:** OOM error with batch_size=8 after disabling gradient checkpointing

**Solution:**
- Reduced batch_size from 8 to 2
- Increased gradient_accumulation from 16 to 64
- Maintained effective batch size of 128

**Result:** Peak VRAM 54GB (67% of 80GB)

### 6.3 Dataset Size Trade-off

**Problem:** Full 666K dataset would require extended training time

**Solution:** Created stratified 200K subsample preserving source ratios

**Result:**
- Reduced training time significantly
- Maintained source distribution

---

## 7. Code & Data Locations

### 7.1 Training Files

**Training Script:**
- **File:** `train_soilfm_cpt_MANUAL.py`
- **Description:** Manual PyTorch training loop with QLoRA, Flash Attention 2, gradient checkpointing disabled
- **Key Modifications:**
  - Line 168: `attn_implementation="flash_attention_2"`
  - Line 183: `prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)`

**Final Training Dataset (200K subsample):**
- **Training File:** `soilfm_train_200k.jsonl`
- **Size:** 200,000 examples
- **Format:** JSONL with fields: `text`, `source`
- **Sources:** PMC (55%), Wikipedia (20%), Soil Survey Manual (10%), Gen Bio (15%)

**Final Validation Dataset (10K subsample):**
- **Validation File:** `soilfm_validation_10k.jsonl`
- **Size:** 10,000 examples
- **Format:** JSONL with fields: `text`, `source`
- **Sources:** Same distribution as training set

**Subsampling Script:**
- **File:** `subsample_dataset.py`
- **Description:** Stratified random sampling (seed=42) from 666K corpus to 200K/10K
- **Input:** Full corpus JSONL files
- **Output:** `soilfm_train_200k.jsonl`, `soilfm_validation_10k.jsonl`

**Documentation Files:**
- **Training Summary:** `SoilFM_Language_Tower_Training_Summary.md`

### 7.2 Training Outputs

**Output Structure:**
```
soilfm-qwen2.5-14b-cpt-200k/
├── best/                      # Best checkpoint (val_loss=1.6130)
│   ├── adapter_model.safetensors (263MB)
│   ├── adapter_config.json
│   └── tokenizer files
├── final/                     # Final checkpoint (same as best)
├── checkpoint-500/            # Intermediate checkpoint
├── checkpoint-1000/           # Intermediate checkpoint
└── checkpoint-1500/           # Intermediate checkpoint
```

**Merged Model:**
- Full model with LoRA merged into base (28GB standalone model)

### 7.3 HuggingFace Repositories (Permanent)

**LoRA Adapter:**
- **URL:** https://huggingface.co/northenlab/soilfm-qwen2.5-14b-qlora
- **Size:** 263MB
- **Contents:** LoRA adapter weights + tokenizer

**Merged Full Model:**
- **URL:** https://huggingface.co/northenlab/soilfm-qwen2.5-14b-literature-cpt
- **Size:** 28GB
- **Contents:** Full model with LoRA merged into base

### 7.4 Experiment Tracking

**Project:** soilfm-language-tower
- **Contents:**
  - Training curves (loss, learning rate, gradient norm)
  - System metrics (GPU utilization, memory)
  - Full training history (1,500 steps)
  - Hyperparameters logged

---

## 8. Reproducibility

### 8.1 Training Command

```bash
python train_qlora_cpt.py \
    --train_file soilfm_train_200k.jsonl \
    --val_file soilfm_validation_10k.jsonl \
    --output_dir ./soilfm-qwen2.5-14b-cpt-200k \
    --epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 64 \
    --use_rslora \
    --use_wandb \
    --wandb_project soilfm-language-tower
```

**Key Configurations:**
1. Flash Attention 2 enabled for performance
2. Gradient checkpointing disabled for speed

### 8.2 Random Seeds

- Dataset subsampling: `random.seed(42)`
- Training: Set in training script

---

## 9. Publication-Ready Information

### 9.1 Key Metrics Table

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-14B-Instruct (14.2B params) |
| Training Examples | 200,000 (stratified) |
| Validation Examples | 10,000 |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| Effective Batch Size | 128 |
| Learning Rate | 2e-5 |
| Training Steps | 1,500 |
| GPU Hours | 42 (A100 80GB) |
| Final Val Loss | 1.6130 |
| Improvement | 7.2% |

### 9.2 Citation

```bibtex
@misc{soilfm-language-tower-2025,
  title={SoilFM Language Tower: Domain Adaptation of Qwen2.5-14B for Soil Science},
  author={Northen Lab},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/northenlab/soilfm-qwen2.5-14b-literature-cpt},
  note={Continued pretraining on 200K soil science literature examples}
}
```

---

**Document Version:** 1.0
**Last Updated:** December 31, 2025
