# DDPM on MNIST

An implementation of Denoising Diffusion Probabilistic Models (DDPM) on the MNIST dataset using PyTorch.

## Overview

This project implements a complete DDPM pipeline with the following features:

- **UNet-mini architecture** without attention layers
- **ε-prediction** objective
- **T=1000** timesteps with linear β schedule
- **Multiple samplers**: Ancestral, Deterministic, and DDIM
- **Comprehensive evaluation**: FID, Inception Score, LPIPS, PSNR
- **Training utilities**: EMA, gradient clipping, checkpointing

## Repository Structure

```
diffusion/
├── model.py          # UNet architecture and time embeddings
├── schedulers.py     # Beta/alpha/sigma schedules, DDIM/EDM
├── losses.py         # Epsilon vs x0 objectives
├── sampler.py        # Ancestral & deterministic samplers
├── conditioning.py   # Class labels, masks, text encoding
├── train.py          # Training loops, EMA, logging
├── sample.py         # Load checkpoint, generate grids
├── utils.py          # EMA, gradient clip, seeding
└── eval.py           # FID/IS/LPIPS/PSNR evaluation

configs/
└── default_config.py # Default configuration

checkpoints/          # Store model checkpoints

data/                 # Dataset storage

samples/              # Store generated images

logs/                 # Store logs from training

evaluation_results/   # Store results from model evaluation (FID/IS/LPIPS/PSNR)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a DDPM model on MNIST:

```bash
python -m diffusion.train --epochs 100 --batch-size 128 --lr 1e-4
```

Resume training from a checkpoint:

```bash
python -m diffusion.train --resume checkpoints/checkpoint_epoch_50.pth --epochs 100 --batch-size 128 --lr 1e-4
```

### Sampling

Generate samples from a trained model:

```bash
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --num-samples 16
```

Generate samples with different samplers:

```bash
# Deterministic sampling (fast)
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --sampler deterministic --steps 1000

# DDIM sampling
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --sampler ddim --steps 50 --eta 0.0

# Ancestral sampling (slow but high quality)
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --sampler ancestral --steps 1000
```

Generate comparison samples:

```bash
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --comparison
```

Generate progressive samples:

```bash
python -m diffusion.sample --checkpoint checkpoints/best_model.pth --progressive
```

### Evaluation

Evaluate model performance:

```bash
python -m diffusion.eval --checkpoint checkpoints/best_model.pth --num-samples 1000
```

## Configuration

The default configuration is in `configs/default_config.py`. Key parameters:

### Model Parameters
- `image_size`: 32 (original MNIST image size is 28x28)
- `model_channels`: 64 (base channel width)
- `time_emb_dim`: 128 (time embedding dimension)

### Training Parameters
- `num_epochs`: 100
- `batch_size`: 128
- `learning_rate`: 1e-4
- `ema_decay`: 0.9999

### Diffusion Parameters
- `num_train_timesteps`: 1000
- `beta_start`: 0.0001
- `beta_end`: 0.02
- `beta_schedule`: 'linear'

## Model Architecture

The UNet-mini architecture consists of:

1. **Time Embedding**: Sinusoidal position embeddings + MLP
2. **Downsampling**: 3 blocks with skip connections
3. **Bottleneck**: 2 conv layers with group norm
4. **Upsampling**: 3 blocks with skip connections
5. **Output**: 1x1 conv for noise prediction

## Training Details

- **Loss**: MSE loss on predicted noise (ε-prediction)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing learning rate
- **Regularization**: EMA with decay 0.9999
- **Gradient Clipping**: Max norm 1.0

## Sampling Methods

1. **Ancestral Sampling**: Standard DDPM sampling (T=1000 steps)
2. **Deterministic Sampling**: DDPM with η=0 (deterministic)
3. **DDIM Sampling**: Fast deterministic sampling with fewer steps

## Evaluation Metrics

- **FID**: Fréchet Inception Distance
- **Inception Score**: Measures image quality and diversity
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **PSNR**: Peak Signal-to-Noise Ratio

## Results

To Add

## License

This project is licensed under the MIT License.# Test commit from dj-shark31 account
