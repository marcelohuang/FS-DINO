#!/usr/bin/env python3
"""Stage 1 evaluation entry point.

Usage:
    python scripts/eval_stage1.py \
        --config configs/stage1_518.yaml \
        --checkpoint checkpoints/stage1_518/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fs_dino.data.dataset import SA1BManifestDataset
from fs_dino.data.transforms import DualResizeTransform
from fs_dino.engine.evaluator import Stage1Evaluator
from fs_dino.models.stage1_distill_model import Stage1DistillModel
from fs_dino.utils.checkpoint import load_weights_only
from fs_dino.utils.logger import setup_logging
from fs_dino.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FS-DINO Stage 1 Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to adapter checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    setup_logging()
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build val dataset
    transform = DualResizeTransform(
        target_res=cfg.data.target_res,
        sam_res=cfg.data.sam_res,
    )
    val_ds = SA1BManifestDataset(
        manifest_path=cfg.data.manifest,
        image_root=cfg.data.get("image_root", ""),
        transform=transform,
        split="val",
        train_frac=cfg.data.train_frac,
        seed=cfg.training.seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Build model and load checkpoint
    model = Stage1DistillModel(cfg).to(device)
    load_weights_only(model.adapter, args.checkpoint, device)

    # Evaluate
    evaluator = Stage1Evaluator(
        model=model,
        dataloader=val_loader,
        device=device,
        align_mode=cfg.model.align_mode,
    )
    results = evaluator.evaluate()

    print("\n--- Stage 1 Evaluation Results ---")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
