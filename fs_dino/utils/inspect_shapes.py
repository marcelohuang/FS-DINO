"""End-to-end shape inspection utility.

Run a single forward pass through the full Stage 1 pipeline and print
all tensor shapes, adapter parameter count, and a sample loss value.
Useful as a post-setup sanity check.

Can work with mock encoders (no real weights) or real weights.
"""

import torch

from fs_dino.engine.losses import alignment_loss
from fs_dino.models.bottleneck_adapter import BottleneckAdapter


def run_shape_check(
    target_res: int = 518,
    batch_size: int = 2,
    device: str = "cpu",
    use_real_models: bool = False,
    cfg=None,
) -> None:
    """Run one forward pass and print all shapes.

    Args:
        target_res: Input image resolution (126, 252, or 518).
        batch_size: Batch size for the dummy tensors.
        device: Device to run on.
        use_real_models: If True, load real DINOv2 + SAM (requires
                         weights). If False, use random tensors.
        cfg: OmegaConf config (required if ``use_real_models=True``).
    """
    patch_size = 14
    feat_h = target_res // patch_size
    feat_w = feat_h

    print(f"{'='*60}")
    print(f"Shape check: target_res={target_res}, batch_size={batch_size}")
    print(f"Expected feature spatial: {feat_h}×{feat_w}")
    print(f"{'='*60}")

    if use_real_models and cfg is not None:
        from fs_dino.models.stage1_distill_model import Stage1DistillModel
        model = Stage1DistillModel(cfg).to(device)
        batch = {
            "dino_image": torch.randn(batch_size, 3, target_res, target_res, device=device),
            "sam_image": torch.randn(batch_size, 3, 1024, 1024, device=device),
        }
        out = model(batch)
        dino_feat = out["dino_feat"]
        adapter_out = out["adapter_out"]
        sam_feat = out["sam_feat"]
        adapter = model.adapter
    else:
        # Mock path: no real weights needed
        dino_feat = torch.randn(batch_size, 768, feat_h, feat_w, device=device)
        sam_feat = torch.randn(batch_size, 256, 64, 64, device=device)
        adapter = BottleneckAdapter().to(device)
        adapter_out = adapter(dino_feat)

    loss = alignment_loss(adapter_out, sam_feat, align_mode="upsample_student")

    # Print shapes
    print(f"  dino_feat:     {tuple(dino_feat.shape)}")
    print(f"  adapter_out:   {tuple(adapter_out.shape)}")
    print(f"  sam_feat:      {tuple(sam_feat.shape)}")
    print(f"  loss:          {loss.item():.6f}")

    # Adapter parameter count
    n_params = sum(p.numel() for p in adapter.parameters())
    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  adapter params: {n_params:,} total, {n_trainable:,} trainable")
    print(f"{'='*60}")


if __name__ == "__main__":
    for res in [126, 252, 518]:
        run_shape_check(target_res=res)
        print()
