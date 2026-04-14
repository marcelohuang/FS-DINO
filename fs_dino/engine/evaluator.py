"""Stage 1 evaluation: MSE loss + cosine similarity diagnostics."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fs_dino.engine.losses import alignment_loss, cosine_similarity_map

logger = logging.getLogger(__name__)


class Stage1Evaluator:
    """Evaluates Stage 1 distillation quality.

    Reports:
      - Mean MSE loss between adapter output and SAM embedding
      - Mean cosine similarity (per spatial location, then averaged)
      - Per-channel MSE statistics (for diagnosing channel-level issues)
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        align_mode: str = "upsample_student",
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.align_mode = align_mode

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run evaluation over the full dataloader.

        Returns:
            dict with ``val_loss``, ``cosine_sim_mean``,
            ``per_channel_mse_mean``, ``per_channel_mse_std``.
        """
        self.model.dino.eval()
        self.model.sam.eval()
        self.model.adapter.eval()

        running_loss = 0.0
        running_cosine = 0.0
        n_samples = 0
        channel_mse_accum = None

        for batch in self.dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            out = self.model(batch)

            student = out["adapter_out"]
            teacher = out["sam_feat"]
            bs = student.shape[0]

            # MSE loss
            loss = alignment_loss(student, teacher, self.align_mode)
            running_loss += loss.item() * bs

            # Cosine similarity (align spatial dims first)
            student_aligned = F.interpolate(
                student, size=(64, 64), mode="bilinear", align_corners=False,
            )
            cos_map = cosine_similarity_map(student_aligned, teacher)  # (B, 64, 64)
            running_cosine += cos_map.mean().item() * bs

            # Per-channel MSE
            per_ch = (student_aligned - teacher).pow(2).mean(dim=(0, 2, 3))  # (256,)
            if channel_mse_accum is None:
                channel_mse_accum = per_ch * bs
            else:
                channel_mse_accum += per_ch * bs

            n_samples += bs

        mean_loss = running_loss / max(n_samples, 1)
        mean_cosine = running_cosine / max(n_samples, 1)
        channel_mse = channel_mse_accum / max(n_samples, 1)

        results = {
            "val_loss": mean_loss,
            "cosine_sim_mean": mean_cosine,
            "per_channel_mse_mean": channel_mse.mean().item(),
            "per_channel_mse_std": channel_mse.std().item(),
        }

        logger.info(
            "Eval | MSE=%.6f | CosSim=%.4f | ChMSE=%.6f±%.6f",
            results["val_loss"],
            results["cosine_sim_mean"],
            results["per_channel_mse_mean"],
            results["per_channel_mse_std"],
        )
        return results
