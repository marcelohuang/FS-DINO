"""Dual-resize transforms for Stage 1 distillation.

Each image is transformed twice:
  1. For DINOv2: resize to target_res, normalise with ImageNet stats.
  2. For SAM:    resize to 1024×1024, normalise with SAM pixel stats.

SAM normalisation note:
  SAM's image encoder expects ``(pixel_255 - mean) / std`` where
  mean=[123.675, 116.28, 103.53] and std=[58.395, 57.12, 57.375] on
  the [0, 255] scale.  After ``ToTensor()`` (which maps to [0, 1]),
  we divide these constants by 255 for use with
  ``torchvision.transforms.Normalize``.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ImageNet statistics (used by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# SAM statistics, converted from [0,255] scale to [0,1] scale
SAM_MEAN = [m / 255.0 for m in [123.675, 116.28, 103.53]]
SAM_STD = [s / 255.0 for s in [58.395, 57.12, 57.375]]


class DualResizeTransform:
    """Produces two tensors from a single PIL image.

    Returns a dict with:
      - ``'dino_image'``: (3, target_res, target_res) ImageNet-normalised
      - ``'sam_image'``:  (3, sam_res, sam_res) SAM-normalised
    """

    def __init__(self, target_res: int = 518, sam_res: int = 1024):
        self.dino_transform = transforms.Compose([
            transforms.Resize(
                (target_res, target_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.sam_transform = transforms.Compose([
            transforms.Resize(
                (sam_res, sam_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=SAM_MEAN, std=SAM_STD),
        ])

    def __call__(self, pil_image):
        return {
            "dino_image": self.dino_transform(pil_image),
            "sam_image": self.sam_transform(pil_image),
        }
