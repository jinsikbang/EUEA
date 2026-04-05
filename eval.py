"""Evaluation script for EUEA (Environmental Understanding Embodied Agent).

Usage
-----
    python eval.py --config configs/eval_config.yaml

Override config values via command line:
    python eval.py --config configs/eval_config.yaml \\
        evaluation.split=test evaluation.batch_size=32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EUEA
from datasets import EmbodiedScanDataset, ScanQADataset, SQA3DDataset
from utils.metrics import compute_metrics
from utils.visualization import visualize_predictions

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


def build_eval_dataloader(cfg, split: str, tokenizer) -> DataLoader:
    dataset_cls = DATASET_REGISTRY[cfg.data.dataset]
    dataset = dataset_cls(cfg.data, split=split)

    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack([b["image"] for b in batch])
        has_pc = batch[0].get("point_cloud") is not None
        point_clouds = (
            torch.stack([b["point_cloud"] for b in batch]) if has_pc else None
        )
        questions = [b["question"] for b in batch]

        # Tokenize questions only (no answer for generation)
        prompts = [f"Q: {q}\nA:" for q in questions]
        encoding = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        result = {
            "images": images,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "questions": questions,
            "answers": [b.get("answers", [b.get("answer", "")]) for b in batch],
            "scene_ids": [b["scene_id"] for b in batch],
            "question_ids": [b.get("question_id", str(i)) for i, b in enumerate(batch)],
        }
        if point_clouds is not None:
            result["point_clouds"] = point_clouds
        return result

    return DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate(
    model: EUEA,
    loader: DataLoader,
    cfg,
    device: torch.device,
    epoch: int | None = None,
    wandb_run=None,
) -> dict[str, float]:
    """Run full evaluation on the provided data loader.

    Args:
        model: EUEA model instance.
        loader: Evaluation data loader.
        cfg: OmegaConf config object.
        device: Torch device.
        epoch: Current training epoch (used for logging).
        wandb_run: Optional W&B run object.

    Returns:
        Dictionary of metric name → score.
    """
    model.eval()
    tokenizer = model.language_decoder.tokenizer
    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_records: list[dict] = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        point_clouds = batch.get("point_clouds")
        if point_clouds is not None:
            point_clouds = point_clouds.to(device)

        generated_ids = model.generate(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )

        # Decode only the newly generated tokens
        input_len = input_ids.shape[1]
        for i, gen_ids in enumerate(generated_ids):
            new_tokens = gen_ids[input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            refs = batch["answers"][i]
            all_predictions.append(pred)
            all_references.append(refs if isinstance(refs, list) else [refs])
            all_records.append({
                "scene_id": batch["scene_ids"][i],
                "question_id": batch["question_ids"][i],
                "question": batch["questions"][i],
                "prediction": pred,
                "answers": refs if isinstance(refs, list) else [refs],
            })

    metrics = compute_metrics(
        all_predictions,
        all_references,
        metrics=list(cfg.evaluation.metrics),
    )

    epoch_str = f"epoch {epoch}" if epoch is not None else "checkpoint"
    logger.info("Evaluation results (%s):", epoch_str)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    if wandb_run is not None:
        log_dict = {f"eval/{k}": v for k, v in metrics.items()}
        if epoch is not None:
            log_dict["epoch"] = epoch
        wandb_run.log(log_dict)

    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)

    if cfg.evaluation.save_predictions:
        pred_path = os.path.join(cfg.evaluation.output_dir, "predictions.json")
        with open(pred_path, "w") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        logger.info("Predictions saved to %s", pred_path)

    metrics_path = os.path.join(cfg.evaluation.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    if cfg.evaluation.visualize:
        vis_dir = os.path.join(cfg.evaluation.output_dir, "visualizations")
        visualize_predictions(all_records, vis_dir)
        logger.info("Visualizations saved to %s", vis_dir)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EUEA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    set_seed(cfg.evaluation.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading model from %s", cfg.evaluation.checkpoint_dir)
    model = EUEA.from_pretrained(cfg.evaluation.checkpoint_dir)
    model.to(device)

    tokenizer = model.language_decoder.tokenizer
    split = cfg.evaluation.split
    loader = build_eval_dataloader(cfg, split, tokenizer)

    # W&B
    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.wandb.get("name", f"eval-{split}"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)

    evaluate(model, loader, cfg, device, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
