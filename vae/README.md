# VAE (Variational Autoencoder) Implementation

This module provides a complete implementation of Variational Autoencoders (VAEs) with CNN architecture for image generation, specifically designed for the MNIST dataset.

## Overview

The VAE implementation includes:

- **CNN-based Encoder/Decoder**: Uses convolutional neural networks for downsampling and upsampling
- **Multiple Loss Functions**: Standard VAE loss, β-VAE loss, and disentangled VAE loss
- **Complete Training Pipeline**: Full training loop with logging, checkpointing, and sampling
- **Sampling Tools**: Various sampling strategies for exploring the latent space
- **Reusable Components**: Leverages functions from the diffusion module for consistency

## Architecture

### Encoder
- **Input**: Images (batch_size, channels, height, width)
- **Process**: Convolutional layers with downsampling (stride=2)
- **Output**: Mean and log variance of latent distribution

### Decoder
- **Input**: Latent vectors (batch_size, latent_dim)
- **Process**: Transposed convolutional layers with upsampling
- **Output**: Reconstructed images

### Key Features
- **Reparameterization Trick**: Enables backpropagation through sampling
- **Batch Normalization**: Stabilizes training
- **LeakyReLU Activation**: Prevents dying ReLU problem
- **Sigmoid Output**: Ensures pixel values in [0, 1] range

## Files Structure

```
vae/
├── __init__.py          # Module initialization
├── model.py             # VAE model architecture
├── losses.py            # Loss functions (VAE, β-VAE, etc.)
├── train.py             # Training pipeline
├── sample.py            # Sampling and generation tools
├── config.py            # Configuration presets
├── example.py           # Usage examples
└── README.md            # This file
```

## Quick Start

### 1. Test the Model

```python
from vae.example import test_vae_model
test_vae_model()
```

### 2. Train a VAE

```python
from vae.example import train_vae_example
model_path = train_vae_example()
```

### 3. Generate Samples

```python
from vae.example import sample_vae_example
sample_vae_example("path/to/model.pth")
```

## Usage

### Training

```python
from vae.train import VAETrainer
from vae.config import get_default_config

# Get configuration
config = get_default_config()
config['experiment_name'] = 'my_vae_experiment'
config['num_epochs'] = 100

# Create trainer and train
trainer = VAETrainer(config)
trainer.train()
```

### Sampling

```python
from vae.sample import VAESampler

# Load trained model
sampler = VAESampler("path/to/model.pth")

# Generate random samples
samples = sampler.sample_random(16)
sampler.save_samples(samples, "samples.png")

# Generate latent grid
grid_samples = sampler.generate_latent_grid(num_samples=8)
sampler.save_latent_grid(grid_samples, "latent_grid.png")

# Interpolate between two points
z1 = torch.randn(1, sampler.model.latent_dim)
z2 = torch.randn(1, sampler.model.latent_dim)
interpolated = sampler.interpolate_latent(z1, z2, 10)
sampler.save_interpolation(interpolated, "interpolation.png")
```

### Command Line Training

```bash
# Train standard VAE
python vae/train.py --experiment_name my_vae --num_epochs 100

# Train β-VAE
python vae/train.py --experiment_name my_beta_vae --loss_type beta_vae --beta_end 4.0

# Train with custom parameters
python vae/train.py \
    --experiment_name custom_vae \
    --latent_dim 64 \
    --hidden_dims 32 64 128 \
    --learning_rate 1e-4 \
    --batch_size 256
```

### Command Line Sampling

```bash
# Generate samples from trained model
python vae/sample.py --model_path checkpoints/my_vae/best_model.pth --output_dir samples

# Generate with custom parameters
python vae/sample.py \
    --model_path checkpoints/my_vae/best_model.pth \
    --output_dir samples \
    --num_samples 25 \
    --grid_size 10 \
    --latent_range_min -4.0 \
    --latent_range_max 4.0
```

## Configuration Options

### Model Parameters
- `in_channels`: Number of input channels (default: 1 for MNIST)
- `hidden_dims`: List of hidden dimensions for encoder/decoder (default: [32, 64, 128, 256])
- `latent_dim`: Dimension of latent space (default: 128)
- `image_size`: Size of input/output images (default: 28)
- `beta`: Weight for KL divergence in loss (default: 1.0)

### Training Parameters
- `batch_size`: Batch size for training (default: 128)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay for regularization (default: 1e-4)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

### Loss Parameters
- `loss_type`: Type of loss ('vae' or 'beta_vae')
- `reconstruction_loss_type`: Type of reconstruction loss ('mse' or 'bce')
- `beta_start`: Starting β value for annealing (default: 0.0)
- `beta_end`: Final β value for annealing (default: 1.0)
- `beta_steps`: Number of steps for β annealing (default: 1000)

## Loss Functions

### Standard VAE Loss
```python
from vae.losses import VAELoss

loss_fn = VAELoss(beta=1.0, loss_type="mse")
total_loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, log_var)
```

### β-VAE Loss
```python
from vae.losses import BetaVAELoss

loss_fn = BetaVAELoss(beta_start=0.0, beta_end=4.0, beta_steps=1000)
# β is automatically updated during training
```

### Disentangled VAE Loss
```python
from vae.losses import DisentangledVAELoss

loss_fn = DisentangledVAELoss(beta=4.0, gamma=1.0)
total_loss, recon_loss, kl_loss, tc_loss = loss_fn(recon_x, x, mu, log_var, z)
```

## Sampling Strategies

### 1. Random Sampling
Generate images by sampling from the prior distribution (standard normal).

### 2. Latent Grid
Create a 2D grid by varying two latent dimensions while keeping others fixed.

### 3. Latent Traversal
Vary one latent dimension while keeping all others fixed to understand what each dimension controls.

### 4. Latent Interpolation
Smoothly interpolate between two points in latent space to see continuous transitions.

## Integration with Diffusion Module

The VAE implementation reuses many functions from the diffusion module:

- **Utility Functions**: `get_device()`, `set_seed()`, `gradient_clip()`, etc.
- **Training Utilities**: `EMA`, `save_checkpoint()`, `load_checkpoint()`, etc.
- **Data Loading**: `create_dataloader()` for consistent data handling
- **Optimization**: `create_optimizer()`, `create_lr_scheduler()` for training setup

This ensures consistency across the codebase and reduces code duplication.

## Examples

### Basic VAE Training
```python
from vae.config import get_default_config
from vae.train import VAETrainer

config = get_default_config()
config['experiment_name'] = 'basic_vae'
trainer = VAETrainer(config)
trainer.train()
```

### β-VAE Training
```python
from vae.config import get_beta_vae_config
from vae.train import VAETrainer

config = get_beta_vae_config()
config['experiment_name'] = 'beta_vae'
trainer = VAETrainer(config)
trainer.train()
```

### Custom Architecture
```python
from vae.model import VAE

# Create custom VAE
vae = VAE(
    in_channels=1,
    hidden_dims=[16, 32, 64, 128],  # Smaller architecture
    latent_dim=64,                  # Smaller latent space
    image_size=28,
    beta=1.0
)
```

## Performance Tips

1. **Batch Size**: Use larger batch sizes (256-512) for better gradient estimates
2. **Learning Rate**: Start with 1e-3 and reduce if training is unstable
3. **β Annealing**: Use β-VAE with annealing for better disentanglement
4. **Regularization**: Use weight decay (1e-4) to prevent overfitting
5. **EMA**: Exponential moving average helps stabilize training

## Troubleshooting

### Common Issues

1. **Training Loss Not Decreasing**
   - Reduce learning rate
   - Increase batch size
   - Check data normalization

2. **Poor Reconstruction Quality**
   - Increase model capacity (more hidden dimensions)
   - Reduce β value to focus on reconstruction
   - Use BCE loss for binary images

3. **Poor Sampling Quality**
   - Increase β value for better latent space
   - Train for more epochs
   - Use β-VAE with annealing

4. **Memory Issues**
   - Reduce batch size
   - Use smaller model architecture
   - Enable gradient checkpointing

## Dependencies

- PyTorch >= 1.9.0
- torchvision
- numpy
- matplotlib
- tqdm
- argparse

## License

This implementation is part of the diffusion model project and follows the same license terms.
