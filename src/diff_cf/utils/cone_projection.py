"""Cone projection from this paper https://arxiv.org/pdf/2210.11841 to ensure use full gradients."""
import math
import torch
import torch.nn.functional as F

from .utils import scalar_product


def cone_projection(grad_seg: torch.Tensor, grad_rob: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Cone projection from this paper https://arxiv.org/pdf/2210.11841 to ensure use full gradients.

    Args:
        grad_seg: gradient of the segmentation model.
        grad_rob: gradient of the robust model.
        alpha: angle in radians.

    Returns:
        new gradient for guiding the diffusion model.
    """
    vec_seg = grad_seg.flatten(1)
    vec_rob = grad_rob.flatten(1)
    angle = torch.acos(F.cosine_similarity(vec_seg, vec_rob, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))    # pylint: disable=not-callable

    p_ortho = vec_rob - (scalar_product(vec_rob, vec_seg) /
                         (scalar_product(vec_seg, vec_seg) + 1e-8)) * vec_seg
    p_ortho_norm = torch.norm(p_ortho, p=2, dim=-1, keepdim=True)
    vec_seg_norm = torch.norm(vec_seg, p=2, dim=-1, keepdim=True)

    u = (math.sin(alpha) * (p_ortho / (p_ortho_norm + 1e-8)) + math.cos(alpha) *
         (vec_seg / (vec_seg_norm + 1e-8)))

    projection = scalar_product(u, vec_rob) * u

    p_cone = torch.where((angle > alpha).unsqueeze(-1), projection, vec_rob)

    return p_cone.view_as(grad_seg)
