"""Stage 1 distillation model: composes DINOv2, SAM teacher, and adapter.

Data flow:
  image → DINOv2 (first 3 blocks) → adapter → student embedding
  image → SAM encoder                       → teacher embedding
  loss = MSE(student_aligned, teacher)

Only the adapter has trainable parameters. DINOv2 and SAM are frozen.
"""

import torch
import torch.nn as nn

from .bottleneck_adapter import BottleneckAdapter
from .dino_encoder import DINOEncoder
from .sam_teacher import SAMTeacher


class Stage1DistillModel(nn.Module):
    """Top-level model for Stage 1 knowledge distillation.

    Wraps three sub-modules but only the adapter is trainable.
    Use ``trainable_parameters()`` (not ``parameters()``) when
    building the optimiser.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: OmegaConf config with ``cfg.model`` containing
                 dino_model, dino_block_idx, dino_norm,
                 sam_checkpoint, sam_model_type.
        """
        super().__init__()
        m = cfg.model

        self.dino = DINOEncoder(
            model_name=m.dino_model,
            block_idx=m.dino_block_idx,
            norm=m.dino_norm,
        )
        self.sam = SAMTeacher(
            checkpoint=m.sam_checkpoint,
            model_type=m.sam_model_type,
        )
        self.adapter = BottleneckAdapter()

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Yields only the adapter parameters (the only trainable ones)."""
        return self.adapter.parameters()

    def trainable_named_parameters(self):
        """Yields (name, param) for adapter parameters only."""
        for name, p in self.adapter.named_parameters():
            yield f"adapter.{name}", p

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys ``'dino_image'`` (B, 3, H, W) and
                   ``'sam_image'`` (B, 3, 1024, 1024).

        Returns:
            dict with:
              - ``'adapter_out'``: (B, 256, H', W') student embedding
              - ``'sam_feat'``:    (B, 256, 64, 64) teacher embedding
              - ``'dino_feat'``:   (B, 768, H', W') raw DINO features
                                   (for debugging / inspection)
        """
        dino_img = batch["dino_image"]
        sam_img = batch["sam_image"]

        # Both encoders are frozen; torch.no_grad() is also inside
        # each encoder's forward, but we add it here for clarity.
        with torch.no_grad():
            dino_feat = self.dino(dino_img)   # (B, 768, H', W')
            sam_feat = self.sam(sam_img)       # (B, 256, 64, 64)

        adapter_out = self.adapter(dino_feat)  # (B, 256, H', W')

        return {
            "adapter_out": adapter_out,
            "sam_feat": sam_feat,
            "dino_feat": dino_feat,
        }
