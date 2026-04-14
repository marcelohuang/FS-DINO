"""Frozen SAM-ViT-H image encoder wrapper for Stage 1.

Loads the SAM-ViT-H image encoder and returns the final embedding
(B×256×64×64) as the distillation target. All parameters are frozen.

The SAM image encoder expects input pre-normalised with SAM's pixel
statistics (handled by the data pipeline in transforms.py).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAMTeacher(nn.Module):
    """Frozen SAM-ViT-H image encoder (teacher).

    Outputs the 256-channel, 64×64 spatial embedding that the
    bottleneck adapter is trained to match via MSE loss.
    """

    def __init__(self, checkpoint: str, model_type: str = "vit_h"):
        """
        Args:
            checkpoint: Path to the SAM checkpoint file
                        (e.g. ``sam_vit_h_4b8939.pth``).
            model_type: SAM model variant. Must be ``"vit_h"`` for Stage 1.
        """
        super().__init__()

        logger.info("Loading SAM model: %s from %s", model_type, checkpoint)
        from segment_anything import sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.encoder = sam.image_encoder

        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        logger.info("SAM image encoder loaded and frozen.")

    # Keep eval mode locked, same rationale as DINOEncoder.
    def train(self, mode: bool = True) -> "SAMTeacher":
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 1024, 1024) SAM-normalised input image tensor.
        Returns:
            (B, 256, 64, 64) image embedding.
        """
        assert x.shape[-2:] == (1024, 1024), (
            f"SAM encoder expects 1024×1024 input, got {x.shape[-2:]}"
        )
        with torch.no_grad():
            return self.encoder(x)  # (B, 256, 64, 64)
