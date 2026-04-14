"""Loss functions for Stage 1 knowledge distillation.

Primary loss: MSE between the adapter output (student) and the SAM
image encoder output (teacher), with spatial alignment via interpolation.
"""

import torch
import torch.nn.functional as F


def alignment_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    align_mode: str = "upsample_student",
) -> torch.Tensor:
    """Compute MSE loss between student and teacher embeddings.

    The student and teacher may have different spatial resolutions.
    ``align_mode`` controls how they are brought to the same size.

    Args:
        student: (B, 256, H_s, W_s) adapter output.
        teacher: (B, 256, H_t, W_t) SAM encoder output (typically 64×64).
        align_mode: ``"upsample_student"`` — bilinear-upsample student to
            teacher size (preserves teacher signal fidelity).
            ``"downsample_teacher"`` — bilinear-downsample teacher to
            student size (for ablation).

    Returns:
        Scalar MSE loss (reduction='mean').
    """
    if align_mode == "upsample_student":
        target_size = (teacher.shape[2], teacher.shape[3])
        student = F.interpolate(
            student, size=target_size, mode="bilinear", align_corners=False,
        )
        return F.mse_loss(student, teacher, reduction="mean")

    elif align_mode == "downsample_teacher":
        target_size = (student.shape[2], student.shape[3])
        teacher = F.interpolate(
            teacher, size=target_size, mode="bilinear", align_corners=False,
        )
        return F.mse_loss(student, teacher, reduction="mean")

    else:
        raise ValueError(f"Unknown align_mode: {align_mode!r}. "
                         f"Use 'upsample_student' or 'downsample_teacher'.")


def cosine_similarity_map(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Compute per-spatial-location cosine similarity.

    Both tensors must have the same spatial size. Returns a (B, H, W)
    map of cosine similarities (useful for evaluation / diagnostics).
    """
    assert student.shape == teacher.shape, (
        f"Shape mismatch: student {student.shape} vs teacher {teacher.shape}"
    )
    # Normalise along channel dim
    s_norm = F.normalize(student, dim=1)
    t_norm = F.normalize(teacher, dim=1)
    # Sum of element-wise product along channel dim → cosine sim
    return (s_norm * t_norm).sum(dim=1)  # (B, H, W)
