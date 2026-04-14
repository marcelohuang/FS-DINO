#!/usr/bin/env python3
"""Prepare SA-1B images from a .tar archive for Stage 1 training.

This script:
  1. Extracts .jpg/.jpeg/.png images from a tar archive into an output directory.
  2. Writes a plain-text manifest (one relative path per line) that is
     directly compatible with SA1BManifestDataset.
  3. Optionally subsamples to the first N or a random N images.
  4. Skips all non-image files (e.g. SA-1B JSON annotations).

After running this script, set these config values:
    data.manifest:    <output_dir>/manifest.txt
    data.image_root:  <output_dir>

Usage examples
--------------
# Extract everything:
    python scripts/prepare_sa1b_from_tar.py \\
        --tar /home/defaultuser/dataset.tar \\
        --out /data/sa1b_images

# Extract only the first 10 000 images (archive order):
    python scripts/prepare_sa1b_from_tar.py \\
        --tar /home/defaultuser/dataset.tar \\
        --out /data/sa1b_images \\
        --first 10000

# Extract a random 10 000 images (reproducible via --seed):
    python scripts/prepare_sa1b_from_tar.py \\
        --tar /home/defaultuser/dataset.tar \\
        --out /data/sa1b_images \\
        --sample 10000 --seed 42

# Verify after extraction (no extraction performed):
    python scripts/prepare_sa1b_from_tar.py \\
        --tar /home/defaultuser/dataset.tar \\
        --out /data/sa1b_images \\
        --verify-only
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import tarfile
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def _collect_image_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Return all TarInfo entries that look like images, sorted by name."""
    members = [m for m in tf.getmembers() if m.isfile() and _is_image(m.name)]
    members.sort(key=lambda m: m.name)
    return members


def _subsample(
    members: list[tarfile.TarInfo],
    first: int | None,
    sample: int | None,
    seed: int,
) -> list[tarfile.TarInfo]:
    if first is not None and sample is not None:
        raise ValueError("--first and --sample are mutually exclusive.")

    if first is not None:
        if first <= 0:
            raise ValueError("--first must be a positive integer.")
        selected = members[:first]
        logger.info("Keeping first %d of %d images.", len(selected), len(members))
        return selected

    if sample is not None:
        if sample <= 0:
            raise ValueError("--sample must be a positive integer.")
        if sample >= len(members):
            logger.info(
                "--sample %d >= total %d; using all images.", sample, len(members)
            )
            return members
        rng = random.Random(seed)
        selected = rng.sample(members, sample)
        # Keep deterministic order for reproducible manifests
        selected.sort(key=lambda m: m.name)
        logger.info(
            "Random-sampled %d of %d images (seed=%d).", len(selected), len(members), seed
        )
        return selected

    return members  # use all


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def extract_and_manifest(
    tar_path: Path,
    out_dir: Path,
    first: int | None,
    sample: int | None,
    seed: int,
) -> Path:
    """Extract images from tar and write a manifest. Returns manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.txt"

    logger.info("Opening tar: %s", tar_path)
    with tarfile.open(tar_path, "r:*") as tf:
        members = _collect_image_members(tf)
        logger.info("Found %d image files in archive.", len(members))

        if not members:
            logger.error(
                "No image files found in %s. "
                "Check that the archive contains .jpg/.jpeg/.png files.",
                tar_path,
            )
            sys.exit(1)

        selected = _subsample(members, first, sample, seed)

        logger.info(
            "Extracting %d images to %s ...", len(selected), out_dir
        )
        rel_paths: list[str] = []
        for i, member in enumerate(selected, 1):
            # Flatten: strip any directory prefix, keep only the filename.
            # SA-1B archives typically have entries like
            #   sa_000001/sa_000001_000000.jpg
            # We preserve the relative path as-is so images stay organised,
            # and the manifest records that same relative path.
            dest = out_dir / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Extract only this one member
            src = tf.extractfile(member)
            if src is None:
                logger.warning("Could not read member %s; skipping.", member.name)
                continue
            dest.write_bytes(src.read())

            rel_paths.append(member.name)

            if i % 1000 == 0 or i == len(selected):
                logger.info("  %d / %d extracted", i, len(selected))

    # Write manifest
    with open(manifest_path, "w") as f:
        f.write("\n".join(rel_paths) + "\n")

    logger.info("Manifest written: %s (%d lines)", manifest_path, len(rel_paths))
    return manifest_path


def verify(out_dir: Path, n_samples: int = 5) -> None:
    """Print basic stats and verify PIL can open a few images."""
    manifest_path = out_dir / "manifest.txt"

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    lines = [l.strip() for l in manifest_path.read_text().splitlines() if l.strip()]
    print(f"\n{'='*60}")
    print(f"Manifest : {manifest_path}")
    print(f"Images   : {len(lines)}")
    print(f"\nFirst {min(n_samples, len(lines))} manifest lines:")
    for line in lines[:n_samples]:
        print(f"  {line}")

    # PIL smoke-test
    try:
        from PIL import Image
    except ImportError:
        print("\n[SKIP] Pillow not installed; skipping PIL open test.")
        return

    print(f"\nPIL open test ({n_samples} images):")
    sample = random.Random(0).sample(lines, min(n_samples, len(lines)))
    all_ok = True
    for rel in sample:
        img_path = out_dir / rel
        try:
            with Image.open(img_path) as im:
                im.verify()  # lightweight — checks header without decoding
            print(f"  OK  {rel}  ({img_path})")
        except Exception as exc:
            print(f"  FAIL {rel}: {exc}")
            all_ok = False

    if all_ok:
        print("\nAll sampled images opened successfully.")
    else:
        print("\nSome images failed to open. Check the extraction.")

    print(f"\nConfig values to set:")
    print(f"  data.manifest:   {manifest_path}")
    print(f"  data.image_root: {out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SA-1B images from a .tar and build a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tar", required=True, type=Path,
        help="Path to the input .tar archive (e.g. /home/defaultuser/dataset.tar).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for extracted images and manifest.txt.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--first", type=int, default=None, metavar="N",
        help="Keep only the first N images (archive order, sorted by name).",
    )
    group.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Keep a random N images. Use --seed for reproducibility.",
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used when --sample is set (default: 42).",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help=(
            "Skip extraction; only run the verification step on an "
            "already-extracted --out directory."
        ),
    )
    parser.add_argument(
        "--verify-samples", type=int, default=5, metavar="N",
        help="Number of images to PIL-open during verification (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.verify_only:
        if not args.tar.exists():
            logger.error("Tar file not found: %s", args.tar)
            sys.exit(1)

        extract_and_manifest(
            tar_path=args.tar,
            out_dir=args.out,
            first=args.first,
            sample=args.sample,
            seed=args.seed,
        )

    verify(args.out, n_samples=args.verify_samples)


if __name__ == "__main__":
    main()
