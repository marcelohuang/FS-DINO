"""Frozen DINOv2-B encoder wrapper for Stage 1.

Loads DINOv2-B (ViT-B/14, patch_size=14) via torch.hub and exposes only
the first 3 transformer blocks. Returns 3rd-layer spatial features in
B×768×H×W format.

Spatial resolution by input size:
  126×126 → 9×9    (126 / 14 = 9)
  252×252 → 18×18  (252 / 14 = 18)
  518×518 → 37×37  (518 / 14 = 37)
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DINOEncoder(nn.Module):
    """Frozen DINOv2-B feature extractor (first 3 blocks only).

    All parameters are frozen. The ``train()`` method is overridden to
    always keep the underlying model in eval mode, preventing accidental
    BatchNorm / dropout mode changes when the parent model calls
    ``model.train()``.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        block_idx: int = 2,
        norm: bool = True,
    ):
        """
        Args:
            model_name: torch.hub model name (default: ``dinov2_vitb14``).
            block_idx: 0-indexed block whose output to extract.
                       block_idx=2 → 3rd transformer block.
            norm: If True, apply the model's final LayerNorm to the
                  intermediate output (stabilises feature magnitudes).
        """
        super().__init__()
        self.block_idx = block_idx
        self.norm = norm

        logger.info("Loading DINOv2 model: %s (block_idx=%d, norm=%s)",
                     model_name, block_idx, norm)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True,
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        logger.info("DINOv2 loaded and frozen. Patch size=%d, embed_dim=%d",
                     self.model.patch_size, self.model.embed_dim)

    # ------------------------------------------------------------------
    # Override train() so the parent training loop cannot flip DINOv2
    # into train mode (which would change LayerNorm / dropout behaviour).
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "DINOEncoder":
        # Always keep the underlying ViT in eval mode
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor, normalised with
               ImageNet statistics.
        Returns:
            (B, 768, H', W') spatial feature map from the 3rd block,
            where H' = H // 14, W' = W // 14.
        """
        with torch.no_grad():
            # get_intermediate_layers returns a list; we request one layer.
            # reshape=True strips CLS and returns B×C×H×W directly.
            feats = self.model.get_intermediate_layers(
                x,
                n=[self.block_idx],
                reshape=True,
                return_class_token=False,
                norm=self.norm,
            )
        return feats[0]  # (B, 768, H', W')
