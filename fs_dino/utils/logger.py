"""Metric logging: console + optional TensorBoard."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger for the project.

    Args:
        log_dir: If provided, also write logs to ``<log_dir>/train.log``.
        level: Logging level.
    """
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / "train.log"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


class MetricLogger:
    """Simple metric logger with optional TensorBoard backend.

    Usage::

        ml = MetricLogger("logs/stage1_126")
        ml.log_scalar("train/loss", 0.42, step=100)
        ml.log_dict({"val/loss": 0.38, "val/cosine_sim": 0.91}, step=200)
        ml.close()
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                logger.warning("tensorboard not installed; logging to console only.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        logger.info("[Step %d] %s: %.6f", step, tag, value)
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict, step: int) -> None:
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
