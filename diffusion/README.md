# DDPM (Denoising Diffusion Probabilistic Models) Module

A complete implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation on the MNIST dataset.

## Overview

This module provides a comprehensive DDPM implementation with the following features:

- **UNet-mini Architecture**: Deep convolutional network without attention layers
- **Multiple Sampling Methods**: Ancestral, Deterministic, and DDIM samplers
- **Comprehensive Evaluation**: FID, Inception Score, LPIPS, PSNR metrics
- **Training Utilities**: EMA, gradient clipping, checkpointing, logging
- **Flexible Configuration**: Customizable diffusion schedules and model parameters
- **Reusable Components**: Utility functions shared across the project

## Architecture

### UNetMini Model
- **Input**: Noisy images + timestep embeddings
- **Architecture**: Downsampling → Bottleneck → Upsampling with skip connections
- **Features**: Group normalization, SiLU activations, time embeddings

### Diffusion Process
- **Timesteps**: T=1000 with linear β schedule
- **Objective**: ε-prediction (predict noise from noisy image)
- **Schedule**: Linear β from 0.0001 to 0.02

### Key Features
- **Skip Connections**: Preserves spatial information
- **Time Embeddings**: Sinusoidal position embeddings + MLP
- **Group Normalization**: Improves training stability
- **SiLU Activations**: Modern activation function for better performance

## Usage

```python
from diffusion.model import UNet
from diffusion.train import DiffusionTrainer
from diffusion.sample import DiffusionSampler

# Create UNet model
model = UNetMini(
    image_size=32,
    in_channels=1
    model_channels=64,
    time_emb_dim=128
)

# Train the model
config = {
    'experiment_name': 'my_diffusion',
    'num_epochs': 100,
    'batch_size': 128,
    # ... other parameters
}
trainer = DiffusionTrainer(config)
trainer.train()

# Sample from trained model
sampler = DiffusionSampler('checkpoints/my_diffusion/best_model.pth')
samples = sampler.sample(num_samples=16)
```

### Command Line Training

```bash
# Train DDPM model
python -m diffusion.train --epochs 100

# Train from customized config
python -m diffusion.train --config configs/config_DDPM.py
```

### Command Line Sampling

```bash
# Generate samples from trained model
python -m diffusion.sample --model_path checkpoints/test_DDPM/best_model.pth --output_dir samples/test_DDPM --num_samples 16
```


## Configuration

### Model Parameters
- `in_channels`: Number of input channels (default: 1 for MNIST)
- `model_channels`: Number of channels in CNN (default: 64)
- `image_size`: Size of input/output images (default: 28)
- `time_emb_dim`: Number of channels of in time-embedding vector

### Training Parameters
- `batch_size`: Batch size for training (default: 128)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay for regularization (default: 1e-4)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `ema_decay`: decay parameter for exponential moving average

### Diffusion Parameters
- `num_train_timesteps`: Number of timesteps in diffusion
- `reconstruction_loss_type`: Type of reconstruction loss ('mse' or 'bce')
- `beta_start`: Starting value of beta
- `beta_end`: Final value of beta
- `beta_schedule`: linear or cosine varaition of beta

## Sampling Methods

### 1. Ancestral Sampling
```python
# Standard DDPM sampling (T=1000 steps)
samples = sampler.sample_ancestral(num_samples=16, steps=1000)
```

### 2. Deterministic Sampling
```python
# DDPM with η=0 (deterministic)
samples = sampler.sample_deterministic(num_samples=16, steps=1000)
```

### 3. DDIM Sampling
```python
# Fast deterministic sampling
samples = sampler.sample_ddim(num_samples=16, steps=50, eta=0.0)
```

### Sampling Quality
- **Ancestral**: Highest quality, slowest
- **Deterministic**: Good quality, moderate speed
- **DDIM**: Fast, good quality with proper η tuning


## Training Features

- **Comprehensive Logging**: Training losses, learning rate scheduling, sample generation
- **EMA (Exponential Moving Average)**: Stabilizes training, improves sample quality
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Regular checkpoints, best model saving, resume capability
- **Multiple Objectives**: ε-prediction and x₀-prediction support

## Integration with Other Modules

The diffusion module provides utility functions used by other modules:

```python
from diffusion.utils import (
    EMA, set_seed, gradient_clip, save_checkpoint, load_checkpoint,
    get_device, log_hyperparameters, create_lr_scheduler, create_optimizer,
    create_dataloader, print_model_summary
)
```

## Evaluation Metrics

### 1. FID (Fréchet Inception Distance)
```python
fid_score = evaluate_fid(generated_samples, real_samples)
```

### 2. Inception Score
```python
inception_score = evaluate_inception_score(generated_samples)
```

### 3. LPIPS (Learned Perceptual Image Patch Similarity)
```python
lpips_score = evaluate_lpips(generated_samples, real_samples)
```

### 4. PSNR (Peak Signal-to-Noise Ratio)
```python
psnr_score = evaluate_psnr(generated_samples, real_samples)
```

## Performance Tips

### Training Stability
- Use EMA with high decay (0.9999)
- Implement gradient clipping
- Use appropriate learning rates (1e-4 for AdamW)
- Monitor training curves carefully

### Model Architecture
- Group normalization improves stability
- SiLU activations for better performance
- Proper time embedding initialization
- Skip connections preserve spatial information


## Troubleshooting

### Common Issues
1. **Poor Sample Quality**: Train longer, check β schedule, use more timesteps
2. **Training Instability**: Reduce learning rate, increase EMA decay, check gradients
3. **Memory Issues**: Reduce batch size, use gradient checkpointing
4. **Slow Sampling**: Use DDIM sampler, reduce timesteps

## 📖 References

1. **DDPM Paper**: "Denoising Diffusion Probabilistic Models"
2. **DDIM Paper**: "Denoising Diffusion Implicit Models"
3. **UNet Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
4. **FID Paper**: "GANs Trained by a Two Time-Scale Update Rule"
