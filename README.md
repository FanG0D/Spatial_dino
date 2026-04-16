# Spatial Residual Encoder

This project implements a **Spatial Residual Encoder** that learns spatial-aware residual features using Spatial Forcing alignment with DUSt3R supervision. Unlike the full SVG autoencoder, this version focuses purely on feature extraction without reconstruction.

## Overview

The Spatial Residual Encoder combines frozen semantic features (DINOv3) with trainable residual features that are aligned to DUSt3R's 3D geometric representations. The key innovation is using Spatial Forcing to implicitly align residual features with DUSt3R's spatial features, enabling the learned latent space to capture both semantic information (from DINOv3) and spatial geometry (from DUSt3R).

## Architecture

```
Input Image
    ├──► Frozen DINOv3 ──────► Semantic Features (384-dim)
    ├──► Frozen DUSt3R ──────► Spatial Features (768-dim) [Supervision]
    └──► Trainable Residual ─► Residual Features (8-dim)
            │
            ▼
    ┌───────────────┐
    │ AlignProjector│ (Spatial Forcing: cosine similarity loss)
    └───────┬───────┘
            ▼
    Distribution Matching
            │
            ▼
    Concat [DINO, Residual] ─► 392-dim Latent (for downstream tasks)
```

**Note**: This is an encoder-only architecture. There is no decoder and no reconstruction loss. The model is trained purely with spatial alignment loss.

## Installation

```bash
# Clone and navigate to the project
cd Spatial_dino

# Install dependencies
pip install -r requirements.txt
```

## Model Components

### 1. Semantic Base: DINOv3 (Frozen)
- Extracts strong semantic features
- 384-dimensional patch tokens
- Completely frozen during training

### 2. Spatial Teacher: DUSt3R (Frozen)
- Provides 3D geometric supervision
- Extracts dense spatial features from images
- Used as target for alignment loss

### 3. Residual Encoder (Trainable)
- Lightweight ViT-S: 6 layers, 8 heads
- 8-dimensional output
- Learns to capture spatial details

### 4. Alignment Projector (Trainable)
- Two-layer MLP with GELU activation
- Projects residual features (8-dim) to DUSt3R dimension (768-dim)
- Implements Spatial Forcing alignment
- Cosine similarity loss for feature alignment

## Training

### Data Preparation

1. Prepare ImageNet dataset:
```bash
# Update data_root in configs/spatial_residual_encoder.yaml
data:
  params:
    train:
      params:
        data_root: /path/to/imagenet
    validation:
      params:
        data_root: /path/to/imagenet
```

2. Prepare DINOv3 weights:
```bash
# Download DINOv3 weights and update config
dinov3_config:
  weights_path: /path/to/dinov3_vits16plus_pretrain.pth
```

3. Prepare DUSt3R weights (optional, for spatial supervision):
```bash
# Download DUSt3R weights and update config
dust3r_config:
  model_path: /path/to/dust3r_checkpoint.pth
```

### Launch Training

```bash
# Single GPU
python main_residual_encoder.py \
    --base configs/spatial_residual_encoder.yaml \
    --train \
    --logdir logs/

# Multi-GPU (8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    main_residual_encoder.py \
    --base configs/spatial_residual_encoder.yaml \
    --train
```

### Resume Training

```bash
# Automatically resumes from latest checkpoint in logdir
python main_residual_encoder.py \
    --base configs/spatial_residual_encoder.yaml \
    --train \
    --logdir logs/spatial_residual
```

## Configuration

Key hyperparameters in `configs/spatial_residual_encoder.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `align_config.loss_coeff` | 0.5 | Weight for spatial alignment loss |
| `align_config.loss_type` | cosine | Loss type: cosine, mse, l1 |
| `align_config.target_dim` | 768 | DUSt3R feature dimension |
| `residual_config.output_dim` | 8 | Residual feature dimension |
| `residual_config.num_layers` | 6 | ViT layers in residual encoder |
| `residual_config.num_heads` | 8 | Number of attention heads |
| `residual_config.hidden_dim` | 384 | Hidden dimension |
| `batch_size` | 64 | Batch size per GPU |
| `base_learning_rate` | 1e-4 | Base learning rate |

## Project Structure

```
Spatial_dino/
├── ldm/
│   ├── models/
│   │   ├── __init__.py
│   │   └── spatial_residual_encoder.py   # Main encoder model
│   ├── modules/
│   │   └── losses/
│   │       └── spatial_forcing_loss.py   # Alignment loss only
│   └── util.py                           # Utilities
├── configs/
│   └── spatial_residual_encoder.yaml     # Configuration
├── main_residual_encoder.py              # Training script
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

## Key Features

1. **Encoder-Only Architecture**: No decoder, no reconstruction loss
2. **Spatial Forcing Alignment**: Implicitly aligns residual features with DUSt3R's 3D representations
3. **Distribution Matching**: Matches residual feature statistics to DINOv3 for stable training
4. **Cosine Similarity Loss**: Encourages directional alignment between features
5. **Frozen Semantic Base**: Preserves DINOv3's strong semantic capabilities

## Usage

### Extract Features

```python
from ldm.models.spatial_residual_encoder import SpatialResidualEncoder

# Load model
model = SpatialResidualEncoder(...)
model.load_state_dict(torch.load("checkpoint.ckpt")["state_dict"])

# Extract features
features = model.extract_features(image)
# Returns dict with:
#   - 'dino': DINOv3 semantic features (384-dim)
#   - 'residual': Raw residual features (8-dim)
#   - 'residual_aligned': Distribution-aligned residual (8-dim)
#   - 'spatial_teacher': DUSt3R features (768-dim)
#   - 'latent': Concatenated [DINO, Residual] (392-dim)
```

## References

- **SVG**: [Latent Diffusion Model without Variational Autoencoder](https://arxiv.org/abs/2510.15301)
- **Spatial Forcing**: [Implicit Spatial Representation Alignment for VLA](https://arxiv.org/abs/2510.12276)
- **DUSt3R**: [Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132)

## Citation

```bibtex
@inproceedings{shi2026svg,
  title={Latent Diffusion Model without Variational Autoencoder},
  author={Shi, Minglei and Wang, Haolin and Zheng, Wenzhao and others},
  booktitle={ICLR},
  year={2026}
}

@article{li2025spatial,
  title={Spatial Forcing: Implicit Spatial Representation Alignment for Vision-Language-Action Model},
  author={Li, Fuhao and Song, Wenxuan and others},
  journal={arXiv preprint arXiv:2510.12276},
  year={2025}
}

@inproceedings{wang2024dust3r,
  title={DUSt3R: Geometric 3D Vision Made Easy},
  author={Wang, Shuzhe and Leroy, Vincent and others},
  booktitle={CVPR},
  year={2024}
}
```
