"""
Wheeler AI - Training Script
============================

Trains a text encoder/decoder where the latent space is a 2D spatial grid
compatible with Wheeler Memory dynamics.

Usage:
    python train.py                     # Basic training
    python train.py --grid_size 32      # Smaller grid (faster)
    python train.py --resume checkpoint.pt  # Resume from checkpoint
"""

import os
import sys
import math
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import our model
from models import WheelerTextAutoencoder, count_parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Train Wheeler Text Autoencoder")
    
    # Model
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--encoder_layers", type=int, default=4, help="Encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=6, help="Decoder layers")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size (64=4K tokens, 32=1K)")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")
    
    # Data
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="Dataset name")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="wheeler-ai", help="W&B project name")
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

class TextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for text data."""
    
    def __init__(self, data, tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # For autoencoding: target = input
        # For language modeling: target = input shifted by 1
        target_ids = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids
        }


def get_dataloaders(args, tokenizer):
    """Load TinyStories dataset and create dataloaders."""
    
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, cache_dir="./data")
    
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    # Optionally limit training samples
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
    
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    
    train_dataset = TextDataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = TextDataset(val_data, tokenizer, args.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    args,
    epoch: int,
    global_step: int,
    device: torch.device,
    wandb_run=None
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=args.fp16):
            # For autoencoding: predict input from grid
            # Shift target for causal LM objective
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            
            logits, grid = model(
                input_ids,
                decoder_input,
                input_mask=attention_mask
            )
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=0  # Assuming 0 is pad token
            )
            
            # Scale for gradient accumulation
            loss = loss / args.accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            optimizer.zero_grad()
            global_step += 1
        
        # Logging
        batch_loss = loss.item() * args.accumulation_steps
        total_loss += batch_loss
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Log to wandb
        if wandb_run and batch_idx % 100 == 0:
            wandb_run.log({
                "train/loss": batch_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": global_step
            })
        
        # Save checkpoint
        if global_step % args.save_every == 0 and global_step > 0:
            save_checkpoint(model, optimizer, scheduler, scaler, args, epoch, global_step)
    
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "global_step": global_step}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    args,
    device: torch.device
) -> Dict[str, float]:
    """Run validation."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits, grid = model(
                input_ids,
                decoder_input,
                input_mask=attention_mask
            )
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=0
            )
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return {"loss": avg_loss, "perplexity": perplexity}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    args,
    epoch: int,
    global_step: int
):
    """Save a checkpoint."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args)
    }
    
    path = os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt")
    torch.save(checkpoint, path)
    print(f"\n✓ Saved checkpoint: {path}")
    
    # Also save as 'latest'
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str
) -> Dict[str, Any]:
    """Load a checkpoint."""
    
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    
    print("=" * 60)
    print("Wheeler AI - Text Autoencoder Training")
    print("=" * 60)
    
    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}")
    
    # Model
    print("\nCreating model...")
    model = WheelerTextAutoencoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        grid_size=args.grid_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Grid size: {args.grid_size}x{args.grid_size} = {args.grid_size**2:,} spatial tokens")
    
    model = model.to(device)
    
    # Optional: torch.compile for speedup
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Data
    train_loader, val_loader = get_dataloaders(args, tokenizer)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, scheduler, scaler, args.resume)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Wandb
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            resume="allow"
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            args, epoch, global_step, device, wandb_run
        )
        global_step = train_metrics["global_step"]
        
        # Validate
        val_metrics = validate(model, val_loader, args, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
        
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_metrics["loss"],
                "val/loss": val_metrics["loss"],
                "val/perplexity": val_metrics["perplexity"]
            })
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, args, epoch, global_step)
            print(f"  ✓ New best model saved!")
    
    # Final save
    save_checkpoint(model, optimizer, scheduler, scaler, args, args.epochs - 1, global_step)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
