"""Stage 1 training loop for knowledge distillation."""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fs_dino.engine.losses import alignment_loss
from fs_dino.utils.checkpoint import save_checkpoint
from fs_dino.utils.logger import MetricLogger

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Trains the bottleneck adapter via MSE distillation.

    Only the adapter parameters are updated. DINOv2 and SAM encoders
    remain frozen throughout.

    Args:
        model: ``Stage1DistillModel`` instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer built from ``model.trainable_parameters()``.
        scheduler: Learning rate scheduler.
        cfg: OmegaConf config.
        metric_logger: ``MetricLogger`` for TensorBoard / console logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg,
        metric_logger: MetricLogger,
        device: torch.device = torch.device("cpu"),
        start_epoch: int = 0,
        global_step: int = 0,
        best_val_loss: float = float("inf"),
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.ml = metric_logger
        self.device = device

        self.start_epoch = start_epoch
        self.global_step = global_step
        self.best_val_loss = best_val_loss

        self.align_mode = cfg.model.align_mode
        self.grad_clip = cfg.training.grad_clip
        self.log_every = cfg.output.log_every
        self.val_every = cfg.output.val_every
        self.save_every = cfg.output.save_every
        self.ckpt_dir = Path(cfg.output.checkpoint_dir)

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop for ``cfg.training.epochs`` epochs."""
        total_epochs = self.cfg.training.epochs

        logger.info(
            "Starting training: epochs %d→%d, %d train / %d val batches",
            self.start_epoch, total_epochs,
            len(self.train_loader), len(self.val_loader),
        )

        for epoch in range(self.start_epoch, total_epochs):
            train_loss = self._train_epoch(epoch)

            # Validation
            if (epoch + 1) % self.val_every == 0 or epoch == total_epochs - 1:
                val_loss = self._validate(epoch)

                self.ml.log_dict({
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/lr": self.optimizer.param_groups[0]["lr"],
                }, step=epoch)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        self.model.adapter, self.optimizer,
                        epoch, self.global_step, val_loss,
                        self.ckpt_dir / "best.pt",
                    )

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                save_checkpoint(
                    self.model.adapter, self.optimizer,
                    epoch, self.global_step, train_loss,
                    self.ckpt_dir / f"epoch_{epoch:03d}.pt",
                )

            self.scheduler.step()

        logger.info("Training complete. Best val loss: %.6f", self.best_val_loss)

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        # Lock frozen encoders in eval; adapter in train
        self.model.dino.eval()
        self.model.sam.eval()
        self.model.adapter.train()

        running_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            out = self.model(batch)
            loss = alignment_loss(out["adapter_out"], out["sam_feat"], self.align_mode)

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.adapter.parameters(), self.grad_clip,
                )

            self.optimizer.step()
            self.global_step += 1

            bs = batch["dino_image"].shape[0]
            running_loss += loss.item() * bs
            n_samples += bs

            if self.global_step % self.log_every == 0:
                self.ml.log_scalar("train/loss_step", loss.item(), self.global_step)

        elapsed = time.time() - t0
        mean_loss = running_loss / max(n_samples, 1)
        logger.info(
            "Epoch %d | train_loss=%.6f | lr=%.2e | %.1fs",
            epoch, mean_loss,
            self.optimizer.param_groups[0]["lr"], elapsed,
        )
        return mean_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> float:
        """Run validation. Returns mean loss."""
        self.model.dino.eval()
        self.model.sam.eval()
        self.model.adapter.eval()

        running_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                out = self.model(batch)
                loss = alignment_loss(out["adapter_out"], out["sam_feat"], self.align_mode)

                bs = batch["dino_image"].shape[0]
                running_loss += loss.item() * bs
                n_samples += bs

        mean_loss = running_loss / max(n_samples, 1)
        logger.info("Epoch %d | val_loss=%.6f", epoch, mean_loss)
        return mean_loss
