"""
Simplified Spatial Forcing Loss Module
Only spatial alignment loss (no reconstruction)
Reference: https://arxiv.org/abs/2510.12276
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialForcingLoss(nn.Module):
    """
    Only spatial alignment loss for Spatial Residual Encoder.
    """

    def __init__(
        self,
        align_loss_type: str = "cosine",
    ):
        super().__init__()
        self.align_loss_type = align_loss_type

    def compute_spatial_alignment_loss(self, f_res, f_dust3r, align_projector):
        """
        Compute spatial alignment loss between residual features and DUSt3R features.
        Reference: Spatial Forcing paper - computes cosine similarity after projection.

        Args:
            f_res: Residual encoder features (B, C_res, N) where N = H*W patches
            f_dust3r: DUSt3R spatial features (B, C_dust3r, N) where N = H*W patches
            align_projector: Projection head that outputs (B, target_dim, N)

        Returns:
            align_loss: Cosine similarity loss
        """
        if f_res is None or f_dust3r is None:
            return torch.tensor(0.0, device=f_res.device if f_res is not None else 'cpu')

        # Project residual features to DUSt3R dimension
        # align_projector: (B, C_res, N) -> (B, target_dim, N)
        f_res_proj = align_projector(f_res)

        # Ensure feature dimensions match (N should already match)
        if f_res_proj.shape[-1] != f_dust3r.shape[-1]:
            # Interpolate along the patch dimension if needed
            f_dust3r = F.interpolate(
                f_dust3r.unsqueeze(-1),  # (B, C, N) -> (B, C, N, 1)
                size=(f_res_proj.shape[-1], 1),
                mode='bilinear',
                align_corners=False
            ).squeeze(-1)  # (B, C, N, 1) -> (B, C, N)

        # Transpose to (B, N, C) for cosine similarity computation
        f_res_flat = f_res_proj.permute(0, 2, 1)  # (B, N, C_res)
        f_dust3r_flat = f_dust3r.permute(0, 2, 1)  # (B, N, C_dust)

        if self.align_loss_type == "cosine":
            # Following Spatial-Forcing: compute_align_loss_cosine
            # L2 normalize both embeddings
            f_res_norm = F.normalize(f_res_flat, dim=-1)
            f_dust3r_norm = F.normalize(f_dust3r_flat, dim=-1)

            # Cosine similarity: (B, N, C) * (B, N, C) -> sum over C -> (B, N)
            cos_sim = (f_res_norm * f_dust3r_norm).sum(dim=-1)  # (B, N)

            # Loss: 1 - cosine_similarity, averaged over batch and patches
            align_loss = (1.0 - cos_sim).mean()

        elif self.align_loss_type == "mse":
            align_loss = F.mse_loss(f_res_flat, f_dust3r_flat)

        elif self.align_loss_type == "l1":
            align_loss = F.l1_loss(f_res_flat, f_dust3r_flat)

        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} not implemented")

        return align_loss

    def forward(
        self,
        f_res,
        f_dust3r,
        align_projector,
        align_coeff=0.5,
        split="train",
    ):
        """
        Forward pass - only alignment loss.

        Args:
            f_res: Residual features
            f_dust3r: DUSt3R features (supervision)
            align_projector: Projection head
            align_coeff: Coefficient for alignment loss
            split: "train" or "val"
        """
        # Spatial alignment loss
        align_loss = self.compute_spatial_alignment_loss(
            f_res, f_dust3r, align_projector
        )

        # Total loss
        loss = align_coeff * align_loss

        # Logging
        log = {
            f"{split}/total_loss": loss.clone().detach(),
            f"{split}/align_loss": align_loss.clone().detach(),
        }

        return loss, log
