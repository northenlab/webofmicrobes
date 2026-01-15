#!/usr/bin/env python3
"""
SoilFM Language Tower: Continued Pretraining (CPT) with QLoRA
MANUAL TRAINING LOOP VERSION

Manual PyTorch training loop with QLoRA for continued pretraining.
Provides full control over training process and memory management.

Usage:
    python train_qlora_cpt.py \
        --train_file soilfm_train.jsonl \
        --val_file soilfm_validation.jsonl \
        --output_dir ./soilfm-qwen2.5-14b-cpt \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation 16 \
        --use_wandb
"""

# CRITICAL: Set environment variables BEFORE importing anything
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import torch
import gc
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from bitsandbytes.optim import PagedAdamW8bit
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="./soilfm-qwen2.5-14b-cpt")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # Lower for CPT vs instruction tuning
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_rslora", action="store_true", help="Use rank-stabilized LoRA")

    # Logging and checkpointing
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="soilfm-language-tower")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def collate_fn(batch, pad_token_id):
    """Collate function - data is already padded to max_seq_length."""
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
    labels = torch.stack([torch.tensor(x['labels']) for x in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def evaluate(model, eval_dataloader, device):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1

    model.train()
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Evaluation complete. Avg loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"soilfm-cpt-manual-r{args.lora_r}",
            config=vars(args)
        )

    logger.info("="*70)
    logger.info("SoilFM Language Tower CPT - Manual Training Loop")
    logger.info("="*70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model with 4-bit quantization
    logger.info("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
    )
    model.config.use_cache = False  # Disable cache for training
    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    # Prepare for LoRA (disable gradient checkpointing for speed)
    logger.info("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # LoRA configuration
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Load datasets
    logger.info("\n" + "="*70)
    logger.info("Loading datasets...")
    logger.info("="*70)

    train_dataset = load_dataset('json', data_files=args.train_file, split='train')
    val_dataset = load_dataset('json', data_files=args.val_file, split='train')

    logger.info(f"Train examples: {len(train_dataset):,}")
    logger.info(f"Val examples: {len(val_dataset):,}")

    # Tokenization function
    def tokenize_fn(examples):
        """Tokenize text and create labels for causal LM."""
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_seq_length,
            padding='max_length',
        )

        # Create labels (replace pad tokens with -100 to ignore in loss)
        labels = []
        for ids in tokenized['input_ids']:
            label = [t if t != tokenizer.pad_token_id else -100 for t in ids]
            labels.append(label)

        tokenized['labels'] = labels
        return tokenized

    # Tokenize datasets
    logger.info("\nTokenizing datasets...")
    columns_to_remove = ["text", "source", "source_id", "chunk_idx", "total_chunks", "tokens"]

    # Remove only columns that exist
    train_cols_to_remove = [c for c in columns_to_remove if c in train_dataset.column_names]
    val_cols_to_remove = [c for c in columns_to_remove if c in val_dataset.column_names]

    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=train_cols_to_remove,
        desc="Tokenizing train"
    )
    val_tokenized = val_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=val_cols_to_remove,
        desc="Tokenizing validation"
    )

    logger.info(f"Tokenization complete")

    # Garbage collection after tokenization
    gc.collect()
    torch.cuda.empty_cache()

    # Create DataLoaders (CRITICAL: num_workers=0 to prevent memory duplication)
    logger.info("\nCreating DataLoaders...")
    train_dataloader = DataLoader(
        train_tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
        num_workers=0,  # CRITICAL: 0 prevents subprocess memory duplication
        pin_memory=False,
    )

    val_dataloader = DataLoader(
        val_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=False,
    )

    logger.info(f"Train batches: {len(train_dataloader):,}")
    logger.info(f"Val batches: {len(val_dataloader):,}")

    # Optimizer
    logger.info("\nSetting up optimizer and scheduler...")
    optimizer = PagedAdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Print training configuration
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Total steps: {total_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Max grad norm: {args.max_grad_norm}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"LoRA dropout: {args.lora_dropout}")
    logger.info(f"Use RSLoRA: {args.use_rslora}")
    logger.info("="*70)
    logger.info("\nMEMORY OPTIMIZATIONS:")
    logger.info("  - 4-bit NF4 quantization")
    logger.info("  - Double quantization enabled")
    logger.info("  - DataLoader num_workers=0 (prevents RAM duplication)")
    logger.info("  - Environment: TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1")
    logger.info("="*70 + "\n")

    # Training loop
    logger.info("Starting training...\n")
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    accumulated_loss = 0
    consecutive_nan = 0  # Track consecutive NaN batches
    max_consecutive_nan = 5  # Stop if too many consecutive NaNs

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*70}")

        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation

            # Check for NaN/Inf with threshold
            if torch.isnan(loss) or torch.isinf(loss):
                consecutive_nan += 1
                logger.warning(f"NaN/Inf loss at step {global_step} (consecutive: {consecutive_nan}/{max_consecutive_nan})")
                if consecutive_nan >= max_consecutive_nan:
                    raise RuntimeError(f"Training stopped: {consecutive_nan} consecutive NaN losses")
                optimizer.zero_grad()
                continue
            else:
                consecutive_nan = 0  # Reset counter on successful batch

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step (after gradient accumulation)
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm
                )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = accumulated_loss * args.gradient_accumulation / args.logging_steps
                    lr = scheduler.get_last_lr()[0]

                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'grad': f'{grad_norm:.2f}'
                    })

                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/grad_norm': grad_norm,
                            'train/epoch': epoch + batch_idx / len(train_dataloader),
                        }, step=global_step)

                    accumulated_loss = 0

                # Evaluation
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, device)
                    logger.info(f"\n  Step {global_step}: val_loss = {val_loss:.4f}")

                    if args.use_wandb:
                        import wandb
                        wandb.log({'eval/loss': val_loss}, step=global_step)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = f"{args.output_dir}/best"
                        logger.info(f"  New best! Saving to {best_dir}")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                    model.train()

                # Checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = f"{args.output_dir}/checkpoint-{global_step}"
                    logger.info(f"\n  Saving checkpoint to {ckpt_dir}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

    # Final save
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    final_dir = f"{args.output_dir}/final"
    logger.info(f"Saving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Final evaluation
    final_val_loss = evaluate(model, val_dataloader, device)
    logger.info(f"\nFinal validation loss: {final_val_loss:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    if args.use_wandb:
        import wandb
        wandb.log({
            'final/val_loss': final_val_loss,
            'final/best_val_loss': best_val_loss,
        })
        wandb.finish()

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
