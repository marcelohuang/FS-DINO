"""Checkpoint save/load utilities for Stage 1 training."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    path: str | Path,
) -> None:
    """Save adapter weights, optimizer state, and training metadata.

    Args:
        adapter: The BottleneckAdapter module.
        optimizer: The optimizer (AdamW).
        epoch: Current epoch (0-indexed).
        global_step: Global training step count.
        val_loss: Current validation loss.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }
    torch.save(state, path)
    logger.info("Checkpoint saved: %s (epoch=%d, val_loss=%.6f)", path, epoch, val_loss)


def load_checkpoint(
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[int, int, float]:
    """Load full checkpoint (adapter + optimizer state).

    Args:
        adapter: The BottleneckAdapter module to load state into.
        optimizer: The optimizer to load state into.
        path: Checkpoint file path.
        device: Device to map tensors to.

    Returns:
        (epoch, global_step, val_loss) from the checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Checkpoint loaded: %s (epoch=%d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["val_loss"]


def load_weights_only(
    adapter: torch.nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> None:
    """Load only the adapter weights (no optimizer state).

    Used for cross-stage weight transfer (e.g. loading best.pt from
    the 126-resolution stage before starting 252-resolution training).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
    logger.info("Adapter weights loaded from: %s", path)
