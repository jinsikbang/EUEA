"""Training script for EUEA (Environmental Understanding Embodied Agent).

Usage
-----
Single-GPU:
    python train.py --config configs/train_config.yaml

Multi-GPU (accelerate):
    accelerate launch train.py --config configs/train_config.yaml

Override config values via command line:
    python train.py --config configs/train_config.yaml \\
        training.batch_size=16 training.num_epochs=5
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EUEA
from datasets import EmbodiedScanDataset, ScanQADataset, SQA3DDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "embodiedscan": EmbodiedScanDataset,
    "scanqa": ScanQADataset,
    "sqa3d": SQA3DDataset,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(cfg, split: str, tokenizer) -> DataLoader:
    dataset_cls = DATASET_REGISTRY[cfg.data.dataset]
    dataset = dataset_cls(cfg.data, split=split)

    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack([b["image"] for b in batch])
        has_pc = batch[0].get("point_cloud") is not None
        point_clouds = (
            torch.stack([b["point_cloud"] for b in batch]) if has_pc else None
        )
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]

        # Tokenize questions + answers as a single sequence for causal LM
        qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
        encoding = tokenizer(
            qa_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        result = {
            "images": images,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }
        if point_clouds is not None:
            result["point_clouds"] = point_clouds
        return result

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


def build_optimizer(model: EUEA, cfg) -> torch.optim.Optimizer:
    # Only optimize trainable parameters (e.g. LoRA weights + projections)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer, cfg, num_training_steps: int):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    num_warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)
    scheduler_name = cfg.training.lr_scheduler

    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(
    model: EUEA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    cfg,
    epoch: int,
    device: torch.device,
    wandb_run=None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        point_clouds = batch.get("point_clouds")
        if point_clouds is not None:
            point_clouds = point_clouds.to(device)

        autocast_dtype = torch.bfloat16 if cfg.training.bf16 else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=cfg.training.bf16 or cfg.training.fp16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                point_clouds=point_clouds,
            )
            loss = outputs["loss"] / cfg.training.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.training.gradient_accumulation_steps

        if (step + 1) % cfg.training.log_every == 0:
            avg = total_loss / (step + 1)
            logger.info(
                "Epoch %d | Step %d/%d | loss=%.4f | lr=%.2e",
                epoch, step + 1, len(loader), avg,
                scheduler.get_last_lr()[0],
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": avg,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": (epoch - 1) * len(loader) + step + 1,
                })

    return total_loss / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EUEA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build model
    model = EUEA(cfg)
    model.to(device)

    tokenizer = model.language_decoder.tokenizer

    # Resume from checkpoint if specified
    start_epoch = 1
    if cfg.training.resume_from:
        logger.info("Resuming from %s", cfg.training.resume_from)
        model = EUEA.from_pretrained(cfg.training.resume_from, cfg)
        model.to(device)
        # Infer starting epoch from directory name convention "epoch_N"
        dirname = os.path.basename(cfg.training.resume_from.rstrip("/"))
        if dirname.startswith("epoch_"):
            start_epoch = int(dirname.split("_")[1]) + 1

    # Data loaders
    train_loader = build_dataloader(cfg, "train", tokenizer)
    val_loader = build_dataloader(cfg, "val", tokenizer)

    # Optimizer and scheduler
    total_steps = (
        len(train_loader) // cfg.training.gradient_accumulation_steps
        * cfg.training.num_epochs
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.fp16)

    # W&B
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.wandb.get("name"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, device, wandb_run,
        )
        logger.info("Epoch %d complete | avg_loss=%.4f", epoch, avg_loss)

        if epoch % cfg.training.save_every == 0:
            ckpt_dir = os.path.join(cfg.training.output_dir, f"epoch_{epoch}")
            model.save_pretrained(ckpt_dir)
            logger.info("Checkpoint saved to %s", ckpt_dir)

        if epoch % cfg.training.eval_every == 0:
            from eval import evaluate
            evaluate(model, val_loader, cfg, device, epoch, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
