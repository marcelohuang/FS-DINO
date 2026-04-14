#!/usr/bin/env python3
"""Stage 1 training entry point.

Usage:
    python scripts/train_stage1.py --config configs/stage1_126.yaml
    python scripts/train_stage1.py --config configs/stage1_252.yaml
    python scripts/train_stage1.py --config configs/stage1_518.yaml

    # Override any config value from CLI:
    python scripts/train_stage1.py --config configs/stage1_126.yaml \
        training.lr=5e-4 training.batch_size=4
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from fs_dino.data.dataset import SA1BManifestDataset
from fs_dino.data.transforms import DualResizeTransform
from fs_dino.engine.trainer import Stage1Trainer
from fs_dino.models.stage1_distill_model import Stage1DistillModel
from fs_dino.utils.checkpoint import load_checkpoint, load_weights_only
from fs_dino.utils.logger import MetricLogger, setup_logging
from fs_dino.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FS-DINO Stage 1 Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*",
                        help="key=value overrides (e.g. training.lr=5e-4)")
    return parser.parse_args()


def build_dataloaders(cfg):
    transform = DualResizeTransform(
        target_res=cfg.data.target_res,
        sam_res=cfg.data.sam_res,
    )

    train_ds = SA1BManifestDataset(
        manifest_path=cfg.data.manifest,
        image_root=cfg.data.get("image_root", ""),
        transform=transform,
        split="train",
        train_frac=cfg.data.train_frac,
        seed=cfg.training.seed,
    )
    val_ds = SA1BManifestDataset(
        manifest_path=cfg.data.manifest,
        image_root=cfg.data.get("image_root", ""),
        transform=transform,
        split="val",
        train_frac=cfg.data.train_frac,
        seed=cfg.training.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader


def build_optimizer_and_scheduler(cfg, model):
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    warmup_epochs = cfg.training.warmup_epochs
    total_epochs = cfg.training.epochs

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return optimizer, scheduler


def main():
    args = parse_args()

    # Load and merge config
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Setup
    setup_logging(cfg.output.log_dir)
    set_seed(cfg.training.seed, deterministic=cfg.training.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build components
    train_loader, val_loader = build_dataloaders(cfg)
    model = Stage1DistillModel(cfg).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    # Resume or cross-stage weight transfer
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    resume_from = cfg.output.get("resume_from", None)
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            # If resuming same-stage training, load optimizer too
            # If cross-stage, only load adapter weights
            try:
                start_epoch, global_step, best_val_loss = load_checkpoint(
                    model.adapter, optimizer, resume_path, device,
                )
                start_epoch += 1  # resume from next epoch
            except KeyError:
                load_weights_only(model.adapter, resume_path, device)

    metric_logger = MetricLogger(cfg.output.log_dir, use_tensorboard=True)

    # Train
    trainer = Stage1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        metric_logger=metric_logger,
        device=device,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
    )
    trainer.train()
    metric_logger.close()


if __name__ == "__main__":
    main()
