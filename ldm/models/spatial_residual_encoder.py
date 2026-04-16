"""
Spatial Residual Encoder with Spatial Forcing
Only extracts spatial-aware residual features (no reconstruction)

Reference:
- SVG: https://github.com/shiml20/SVG
- Spatial Forcing: https://github.com/haofuly/spatial-forcing
- DUSt3R: https://github.com/naver/dust3r
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple

from ldm.util import instantiate_from_config


def create_small_vit_s(
    output_dim: int = 8,
    patch_size: int = 16,
    img_size: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 384,
    mlp_dim: int = 1536,
    dropout: float = 0.1,
):
    """
    Create a lightweight ViT-S model as Residual Encoder.
    Based on SVG paper's implementation.
    """
    from torchvision.models.vision_transformer import VisionTransformer

    num_patches = (img_size // patch_size) ** 2

    vit_config = {
        'image_size': img_size,
        'patch_size': patch_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'hidden_dim': hidden_dim,
        'mlp_dim': mlp_dim,
        'num_classes': output_dim,
        'dropout': dropout,
        'attention_dropout': dropout,
    }

    model = VisionTransformer(**vit_config)

    # Replace classification head with projection to output_dim
    model.heads = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    )

    # Custom forward to return patch-level features
    # Following SVG: returns (B, output_dim, num_patches) = (B, 8, 256)
    def forward_custom(x):
        x = model._process_input(x)
        batch_size = x.shape[0]

        # Add class token
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Pass through Transformer encoder
        x = model.encoder(x)

        # Remove class token, keep patch tokens only
        x = x[:, 1:, :]  # shape: (B, num_patches, hidden_dim)

        # Apply head projection
        x = model.heads(x)  # shape: (B, num_patches, output_dim)

        # Transpose to (B, output_dim, num_patches) - matching SVG exactly
        return x.transpose(1, 2)  # (B, 8, 256)

    model.forward = forward_custom
    return model


class AlignProjector(nn.Module):
    """
    Alignment projector for Spatial Forcing.
    Projects residual features to match DUSt3R feature dimensions.
    Reference: Spatial Forcing paper (projectors.py)

    Architecture: Two-layer MLP with GELU activation
    - fc1: residual_dim -> 2 * target_dim
    - fc2: 2 * target_dim -> target_dim

    Input/Output format: (B, C, N) where N = H*W (number of patches)
    """

    def __init__(
        self,
        residual_dim: int,
        target_dim: int,
        use_norm: bool = True,
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.target_dim = target_dim
        hidden_dim = 2 * target_dim

        # Two-layer MLP with GELU activation
        self.fc1 = nn.Linear(residual_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, target_dim, bias=True)
        self.act_fn = nn.GELU()

        # Optional LayerNorm (applied to residual features before projection)
        self.norm = nn.LayerNorm(residual_dim) if use_norm else None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        # x: (B, C, N) where N = H * W (e.g., 256 for 16x16 patches)
        x = x.permute(0, 2, 1)  # (B, N, C)

        if self.norm is not None:
            x = self.norm(x)

        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)  # (B, N, target_dim)

        # Transpose back to (B, target_dim, N)
        x = x.permute(0, 2, 1)  # (B, target_dim, N)
        return x


class DUSt3RWrapper(nn.Module):
    """
    Wrapper for DUSt3R model to extract spatial features.
    Keeps the model frozen and extracts pointmap-based features.
    """

    def __init__(self, model_path: Optional[str] = None, dust3r_root: Optional[str] = None):
        super().__init__()
        self.model_path = model_path
        self.dust3r_root = dust3r_root or "/mnt/disk_2/deyi/dust3r"
        self.model = None
        self._load_model()

        # Freeze all parameters
        if self.model is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        """Load DUSt3R model."""
        try:
            # Add dust3r to path if needed
            if self.dust3r_root not in sys.path:
                sys.path.insert(0, self.dust3r_root)

            from dust3r.model import AsymmetricCroCo3DStereo

            if self.model_path:
                self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path)
            else:
                raise ValueError("model_path must be provided for DUSt3R")

        except Exception as e:
            print(f"Warning: Could not load DUSt3R model: {e}")
            print("Using placeholder for DUSt3R features")
            self.model = None

    def forward(self, x):
        """
        Extract spatial features from input image.
        Args:
            x: (B, 3, H, W) tensor in [0, 1] range
        Returns: (B, 768, N) where N = H*W patches (DUSt3R ViT-L dimension)
        """
        if self.model is None:
            B, C, H, W = x.shape
            N = (H // 16) * (W // 16)
            return torch.randn(B, 768, N, device=x.device)

        with torch.no_grad():
            # Compute true_shape for DUSt3R
            B, C, H, W = x.shape
            true_shape = torch.tensor([[H, W]] * B, device=x.device)

            # Use _encode_image to get features
            feat, _, _ = self.model._encode_image(x, true_shape)
            # feat: (B, N, 768) -> transpose to (B, 768, N)
            feat = feat.transpose(1, 2)

        return feat


class DINOv3Wrapper(nn.Module):
    """
    Wrapper for DINOv3 model to extract semantic features.
    Keeps the model frozen as the semantic base.
    """

    def __init__(self, model_name: str = "dinov3_vits16plus", weights_path: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.weights_path = weights_path
        self.model = None
        self.feature_dim = 384  # DINOv3 ViT-S feature dimension
        self.patch_size = 16
        self.num_reg = 0  # Number of register tokens
        self._load_model()

    def _load_model(self):
        """Load DINOv3 model using transformers.AutoModel."""
        try:
            from transformers import AutoModel

            if self.weights_path:
                self.model = AutoModel.from_pretrained(self.weights_path)
            else:
                raise ValueError("weights_path must be provided for DINOv3")

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            # Get number of register tokens (DINOv2/v3 have registers)
            self.num_reg = getattr(self.model.config, 'num_register_tokens', 0)

        except Exception as e:
            print(f"Warning: Could not load DINOv3 model: {e}")
            print("Using placeholder for DINOv3 features")
            self.model = None

    def forward(self, x):
        """Extract semantic features from input image.
        Args:
            x: (B, 3, H, W) tensor, ImageNet normalized
        Returns: (B, feature_dim, N) where N = H*W (e.g., 196 for 224x224 with 16x16 patches)
        """
        if self.model is None:
            B, _, H, W = x.shape
            N = (H // 16) * (W // 16)
            return torch.randn(B, self.feature_dim, N, device=x.device)

        with torch.no_grad():
            # Use transformers API
            outputs = self.model(pixel_values=x)
            # last_hidden_state: (B, 1+num_reg+N, D) where 1 is cls token
            features = outputs.last_hidden_state

            # Remove cls token and register tokens: [:, 1+num_reg:, :]
            features = features[:, 1 + self.num_reg:, :]  # (B, N, D)

            # Transpose to (B, D, N) format to match SVG
            features = features.transpose(1, 2)  # (B, feature_dim, N)

        return features


class SpatialResidualEncoder(pl.LightningModule):
    """
    Spatial Residual Encoder with Spatial Forcing.
    Only learns residual features, no reconstruction.

    Combines:
    1. Frozen DINOv3 for semantic features
    2. Frozen DUSt3R for spatial supervision
    3. Trainable Residual Encoder with Spatial Forcing alignment
    """

    def __init__(
        self,
        lossconfig: Dict,
        dinov3_config: Dict,
        dust3r_config: Dict,
        residual_config: Dict,
        align_config: Dict,
        image_key: str = "image",
        monitor: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        ignore_keys: list = [],
    ):
        super().__init__()
        self.image_key = image_key

        # Initialize encoders
        self.semantic_encoder = DINOv3Wrapper(**dinov3_config)
        self.spatial_teacher = DUSt3RWrapper(**dust3r_config)

        # Residual encoder (trainable)
        self.residual_encoder = create_small_vit_s(**residual_config)

        # Alignment projector
        self.align_projector = AlignProjector(
            residual_dim=residual_config.get('output_dim', 8),
            target_dim=align_config.get('target_dim', 768),
            use_norm=align_config.get('use_norm', True),
        )

        # Loss module (only spatial forcing loss)
        self.loss = instantiate_from_config(lossconfig)

        # Spatial forcing coefficient
        self.align_coeff = align_config.get('loss_coeff', 0.5)

        # For logging
        if monitor is not None:
            self.monitor = monitor

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path: str, ignore_keys: list = []):
        """Load checkpoint."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def match_distribution(self, h_res: torch.Tensor, h_dino: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Match residual feature distribution to DINO feature distribution.
        Reference: SVG paper Equation (6) and dino_decoder.py match_distribution()

        Following SVG implementation:
        1. Compute mean and std across batch and spatial dimensions
        2. Average across channels to get scalar statistics
        3. Normalize residual features and rescale to DINO distribution

        Args:
            h_res: Residual features (B, C, N) where N = H*W
            h_dino: DINO features (B, C_dino, N) where N = H*W
        """
        # Compute global mean and std for DINO features (per-channel)
        # For 3D input (B, C, N), we average over batch (0) and spatial (2) dims
        mean_dino = h_dino.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
        std_dino = h_dino.std(dim=(0, 2), keepdim=True)   # (1, C, 1)

        # Average across channels to get scalar statistics (following SVG)
        mean_dino_scalar = mean_dino.mean().detach()  # scalar
        std_dino_scalar = std_dino.mean().detach()    # scalar

        # Compute mean and std for residual features (per-channel)
        mean_res = h_res.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
        std_res = h_res.std(dim=(0, 2), keepdim=True)    # (1, C, 1)

        # Average across channels to get scalar statistics (following SVG)
        mean_res_scalar = mean_res.mean().detach()  # scalar
        std_res_scalar = std_res.mean().detach()    # scalar

        # Normalize and re-scale using scalar statistics
        h_res_normed = (h_res - mean_res_scalar) / (std_res_scalar + eps)
        h_res_aligned = h_res_normed * std_dino_scalar + mean_dino_scalar

        return h_res_aligned

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.

        Returns:
            f_dino: DINO semantic features (B, 384, N) where N = H*W patches
            f_res: Residual features (B, 8, N) where N = H*W patches
            f_res_aligned: Distribution-aligned residual features (B, 8, N)
            f_dust3r: DUSt3R spatial features for supervision (B, D, N)
        """
        # Extract semantic features (frozen)
        f_dino = self.semantic_encoder(x)

        # Extract spatial supervision features (frozen)
        f_dust3r = self.spatial_teacher(x)

        # Extract residual features (trainable)
        f_res = self.residual_encoder(x)

        # Match distribution to DINO
        f_res_aligned = self.match_distribution(f_res, f_dino)

        return f_dino, f_res, f_res_aligned, f_dust3r

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        return self.encode(input)
    
    def get_input(self, batch: Dict, k: str) -> torch.Tensor:
        """
        Prepare input batch.
        Returns normalized image for DINO and DUSt3R.
        """
        x = batch[k]  # Shape: (B, 3, H, W) from AirSim dataset

        # x is already in (B, C, H, W) format from dataset's ToTensor()
        # No need to permute

        # Ensure float type
        x = x.float()

        # Normalize for DINO (ImageNet normalization)
        # Convert from [-1, 1] to [0, 1] if needed
        if x.min() < 0:  # Data is in [-1, 1] range
            x_dino = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        else:
            x_dino = x  # Already in [0, 1]

        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_dino = (x_dino - mean) / std

        return x_dino

    def training_step(self, batch: Dict, batch_idx: int):
        """Training step with Spatial Forcing (no reconstruction)."""
        # Get input
        inputs_dino = self.get_input(batch, self.image_key)

        # Forward pass
        f_dino, f_res, f_res_aligned, f_dust3r = self(inputs_dino)

        # Compute spatial alignment loss
        loss, log_dict = self.loss(
            f_res=f_res,
            f_dust3r=f_dust3r,
            align_projector=self.align_projector,
            align_coeff=self.align_coeff,
            split="train",
        )

        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(log_dict, prog_bar=False)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        inputs_dino = self.get_input(batch, self.image_key)
        f_dino, f_res, f_res_aligned, f_dust3r = self(inputs_dino)

        loss, log_dict = self.loss(
            f_res=f_res,
            f_dust3r=f_dust3r,
            align_projector=self.align_projector,
            align_coeff=self.align_coeff,
            split="val",
        )

        self.log("val/loss", loss, prog_bar=True)
        self.log_dict(log_dict, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """Configure optimizers (only residual encoder and projector)."""
        lr = self.learning_rate

        # Optimizer for residual encoder and alignment projector
        params = (
            list(self.residual_encoder.parameters()) +
            list(self.align_projector.parameters())
        )

        opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        return opt

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all features for downstream use.
        Returns dict with keys: 'dino', 'residual', 'residual_aligned', 'spatial_teacher', 'latent'
        All features have format (B, C, N) where N = H*W (number of patches)
        """
        f_dino, f_res, f_res_aligned, f_dust3r = self.encode(x)

        # Concatenate final latent representation
        latent = torch.cat([f_dino, f_res_aligned], dim=1)

        return {
            'dino': f_dino,
            'residual': f_res,
            'residual_aligned': f_res_aligned,
            'spatial_teacher': f_dust3r,
            'latent': latent,
        }
