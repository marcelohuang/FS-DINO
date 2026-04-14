"""Manifest-based image dataset for Stage 1 distillation.

Supports .txt (one path per line), .csv (first column), or .json
(list of strings) manifests. The dataset is image-only — no labels
are needed for Stage 1.

The manifest + image_root design makes it easy to swap SA-1B for
custom domain images (e.g. rebar) by just changing the manifest file.
"""

import csv
import json
import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SA1BManifestDataset(Dataset):
    """Image-only dataset driven by a manifest file.

    Args:
        manifest_path: Path to .txt, .csv, or .json manifest.
        image_root: Root directory prepended to relative paths in the
                    manifest. If paths in the manifest are absolute,
                    set this to ``""``.
        transform: Callable that takes a PIL image and returns a dict
                   (e.g. ``DualResizeTransform``).
        split: ``"train"`` or ``"val"``.
        train_frac: Fraction of data for the train split.
        seed: Random seed for deterministic splitting.
    """

    def __init__(
        self,
        manifest_path: str,
        image_root: str = "",
        transform=None,
        split: str = "train",
        train_frac: float = 0.9,
        seed: int = 42,
    ):
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.split = split

        all_paths = self._parse_manifest(manifest_path)
        logger.info("Manifest %s: %d total images", manifest_path, len(all_paths))

        # Deterministic split
        rng = random.Random(seed)
        indices = list(range(len(all_paths)))
        rng.shuffle(indices)

        n_train = int(len(all_paths) * train_frac)
        if split == "train":
            self.paths = [all_paths[i] for i in indices[:n_train]]
        elif split == "val":
            self.paths = [all_paths[i] for i in indices[n_train:]]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        logger.info("Split '%s': %d images", split, len(self.paths))

    # ------------------------------------------------------------------
    # Manifest parsing
    # ------------------------------------------------------------------
    def _parse_manifest(self, path: str) -> list[str]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]

        elif suffix == ".csv":
            with open(path) as f:
                reader = csv.reader(f)
                return [row[0].strip() for row in reader if row]

        elif suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(p) for p in data]
            raise ValueError("JSON manifest must be a list of path strings")

        else:
            raise ValueError(f"Unsupported manifest format: {suffix}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            return p
        if self.image_root is not None:
            return self.image_root / p
        return p

    def __getitem__(self, idx: int) -> dict:
        img_path = self._resolve_path(self.paths[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load %s: %s. Returning random sample.", img_path, e)
            return self[random.randint(0, len(self) - 1)]

        if self.transform is not None:
            return self.transform(image)

        return {"image": image}
