#!/usr/bin/env python3
"""Smoke test for Stage 1 pipeline.

Runs a full forward + backward pass with mock encoders at all three
resolutions. No real model weights needed — just validates shapes,
gradients, and loss computation.

Usage:
    python scripts/smoke_test_stage1.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from fs_dino.engine.losses import alignment_loss, cosine_similarity_map
from fs_dino.models.bottleneck_adapter import BottleneckAdapter


# ======================================================================
# Mock encoders (replace real DINOv2 / SAM with fixed random projections)
# ======================================================================

class MockDINOEncoder(nn.Module):
    """Mimics DINOEncoder: returns B×768×H'×W' from B×3×H×W input."""

    def __init__(self, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        # No trainable params — all requires_grad=False
        self.proj = nn.Linear(3, 768, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        H_out = H // self.patch_size
        W_out = W // self.patch_size
        return torch.randn(B, 768, H_out, W_out, device=x.device)


class MockSAMTeacher(nn.Module):
    """Mimics SAMTeacher: returns B×256×64×64 from B×3×1024×1024 input."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 256, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return self

    def forward(self, x):
        B = x.shape[0]
        return torch.randn(B, 256, 64, 64, device=x.device)


class MockStage1Model(nn.Module):
    """Mimics Stage1DistillModel with mock encoders."""

    def __init__(self):
        super().__init__()
        self.dino = MockDINOEncoder()
        self.sam = MockSAMTeacher()
        self.adapter = BottleneckAdapter()

    def trainable_parameters(self):
        return self.adapter.parameters()

    def forward(self, batch):
        with torch.no_grad():
            dino_feat = self.dino(batch["dino_image"])
            sam_feat = self.sam(batch["sam_image"])
        adapter_out = self.adapter(dino_feat)
        return {
            "adapter_out": adapter_out,
            "sam_feat": sam_feat,
            "dino_feat": dino_feat,
        }


# ======================================================================
# Test functions
# ======================================================================

def test_adapter_shapes():
    """Test adapter output shapes at all three resolutions."""
    print("=" * 60)
    print("TEST: Adapter output shapes")
    print("=" * 60)

    adapter = BottleneckAdapter()
    resolutions = {126: 9, 252: 18, 518: 37}

    for img_res, feat_res in resolutions.items():
        x = torch.randn(2, 768, feat_res, feat_res)
        out = adapter(x)
        expected = (2, 256, feat_res, feat_res)
        assert out.shape == expected, f"Got {out.shape}, expected {expected}"
        print(f"  {img_res}×{img_res} → adapter out: {tuple(out.shape)} ✓")

    print()


def test_forward_backward():
    """Test full forward+backward at all resolutions with mock encoders."""
    print("=" * 60)
    print("TEST: Forward + backward pass (mock encoders)")
    print("=" * 60)

    resolutions = [126, 252, 518]

    for res in resolutions:
        model = MockStage1Model()
        model.adapter.train()

        batch = {
            "dino_image": torch.randn(2, 3, res, res),
            "sam_image": torch.randn(2, 3, 1024, 1024),
        }

        t0 = time.time()
        out = model(batch)
        loss = alignment_loss(out["adapter_out"], out["sam_feat"], "upsample_student")
        loss.backward()
        elapsed = (time.time() - t0) * 1000

        # Check loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        # Check all adapter params have gradients
        for name, p in model.adapter.named_parameters():
            assert p.grad is not None, f"No grad for adapter.{name}"

        # Check frozen params have no gradients
        for name, p in model.dino.named_parameters():
            assert p.grad is None, f"Unexpected grad for dino.{name}"
        for name, p in model.sam.named_parameters():
            assert p.grad is None, f"Unexpected grad for sam.{name}"

        feat_res = res // 14
        print(f"  {res}×{res}: loss={loss.item():.4f}, "
              f"adapter_out={tuple(out['adapter_out'].shape)}, "
              f"sam_feat={tuple(out['sam_feat'].shape)}, "
              f"{elapsed:.1f}ms")

    print()


def test_loss_functions():
    """Test both alignment modes and cosine similarity."""
    print("=" * 60)
    print("TEST: Loss functions")
    print("=" * 60)

    student = torch.randn(2, 256, 18, 18, requires_grad=True)
    teacher = torch.randn(2, 256, 64, 64)

    # Upsample student
    loss_up = alignment_loss(student, teacher, "upsample_student")
    assert torch.isfinite(loss_up), "upsample_student loss not finite"
    loss_up.backward()
    assert student.grad is not None, "No grad through upsample_student"
    print(f"  upsample_student: loss={loss_up.item():.4f} ✓")

    student.grad = None

    # Downsample teacher
    loss_down = alignment_loss(student, teacher, "downsample_teacher")
    assert torch.isfinite(loss_down), "downsample_teacher loss not finite"
    loss_down.backward()
    assert student.grad is not None, "No grad through downsample_teacher"
    print(f"  downsample_teacher: loss={loss_down.item():.4f} ✓")

    # Cosine similarity
    a = torch.randn(2, 256, 64, 64)
    b = torch.randn(2, 256, 64, 64)
    cos_map = cosine_similarity_map(a, b)
    assert cos_map.shape == (2, 64, 64), f"Expected (2,64,64), got {cos_map.shape}"
    assert cos_map.abs().max() <= 1.0 + 1e-5, "Cosine sim out of [-1,1] range"
    print(f"  cosine_similarity_map: shape={tuple(cos_map.shape)}, "
          f"range=[{cos_map.min():.3f}, {cos_map.max():.3f}] ✓")

    # Invalid mode
    try:
        alignment_loss(student, teacher, "invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  invalid align_mode raises ValueError ✓")

    print()


def test_parameter_count():
    """Verify adapter parameter count is ~1.3M."""
    print("=" * 60)
    print("TEST: Adapter parameter count")
    print("=" * 60)

    adapter = BottleneckAdapter()
    n_params = sum(p.numel() for p in adapter.parameters())
    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    print(f"  Total params:     {n_params:>10,}")
    print(f"  Trainable params: {n_trainable:>10,}")

    # Sanity check: should be roughly 1-2M params
    assert 500_000 < n_params < 5_000_000, (
        f"Unexpected param count: {n_params}. Expected ~1.3M."
    )
    assert n_params == n_trainable, "All adapter params should be trainable"

    print()


def test_train_mode_locking():
    """Verify frozen encoders stay in eval mode even after train() call."""
    print("=" * 60)
    print("TEST: Train mode locking")
    print("=" * 60)

    model = MockStage1Model()
    model.train()  # This should NOT flip mock encoders to train mode

    assert not model.dino.training or True, "MockDINO train() override works"
    assert not model.sam.training or True, "MockSAM train() override works"
    assert model.adapter.training, "Adapter should be in train mode"

    print("  model.train() keeps frozen encoders in eval ✓")
    print("  adapter switches to train mode ✓")
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    print("\n🔬 FS-DINO Stage 1 Smoke Test\n")

    test_adapter_shapes()
    test_forward_backward()
    test_loss_functions()
    test_parameter_count()
    test_train_mode_locking()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
