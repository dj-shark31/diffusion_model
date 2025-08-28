# GAN (Generative Adversarial Network) Module

A complete implementation of Generative Adversarial Networks (GANs) with DCGAN architecture for image generation on the MNIST dataset.

## 🎯 Overview

This module provides a comprehensive GAN implementation with the following features:

- **DCGAN Architecture**: Deep Convolutional GAN with transposed convolutions for generator and convolutional layers for discriminator
- **Multiple Loss Functions**: Standard GAN, Wasserstein GAN (WGAN), and Hinge loss
- **Feature Matching**: Optional feature matching loss for improved training stability
- **Comprehensive Training**: Full training pipeline with logging, checkpointing, and sampling
- **Latent Space Exploration**: Tools for interpolation, traversal, and grid generation

## 📁 Module Structure

```
gan/
├── __init__.py          # Module initialization
├── model.py            # GAN model architecture (Generator, Discriminator, GAN)
├── losses.py           # Loss functions (GANLoss, WassersteinLoss, HingeLoss, FeatureMatchingLoss)
├── train.py            # Training pipeline (GANTrainer)
├── sample.py           # Sampling and generation (GANSampler)
└── README.md           # This documentation
```

## 🏗️ Architecture

### Generator
- **Input**: Random noise vector (latent_dim)
- **Architecture**: Linear projection → Transposed convolutions → Tanh output
- **Features**: Group normalization, SiLu activations, progressive upsampling

### Discriminator
- **Input**: Images (channels × height × width)
- **Architecture**: Convolutional layers → Flatten → Linear → Sigmoid
- **Features**: Group normalization, LeakyReLU activations, progressive downsampling

### Key Features
- **DCGAN Design**: Follows DCGAN paper architecture principles
- **Batch Normalization**: Improves training stability
- **LeakyReLU**: Prevents gradient vanishing in discriminator
- **Tanh Output**: Generator outputs in [-1, 1] range
- **Sigmoid Output**: Discriminator outputs probabilities in [0, 1]

## Usage

```python
from gan.model import GAN
from gan.train import GANTrainer
from gan.sample import GANSampler

### Training
gan = GAN(
    latent_dim=128,
    hidden_dims=[512, 256, 128],
    in_channels=1,
    image_size=32
)

# Train the model
config = {
    'experiment_name': 'my_gan',
    'num_epochs': 100,
    'batch_size': 64,
    # ... other parameters
}
trainer = GANTrainer(config)
trainer.train()

# Sample from trained model
sampler = GANSampler('checkpoints/my_gan/best_model.pth')
samples = sampler.sample_random(num_samples=16)
```

### Command Line Training

```bash
# Train standard GAN
python -m gan.train --epochs 100

# Train from customized config
python -m gan.train --config configs/config_GAN.py
```

### Command Line Sampling

```bash
# Generate samples from trained model
python -m gan.sample --model_path checkpoints/test_VAE/best_model.pth --output_dir samples/test_GAN --num_samples 16
```

## Loss Functions

### 1. Standard GAN Loss
```python
from gan.losses import GANLoss

loss_fn = GANLoss(label_smoothing=0.1)
d_loss, g_loss = loss_fn(real_probs, fake_probs)
```

**Features:**
- Binary cross entropy loss
- Label smoothing for stability
- Separate losses for generator and discriminator

### 2. Wasserstein Loss
```python
from gan.losses import WassersteinLoss

loss_fn = WassersteinLoss()
d_loss, g_loss = loss_fn(real_scores, fake_scores)
```

**Features:**
- Wasserstein distance approximation
- Uses raw scores (before sigmoid)
- Better training stability

### 3. Hinge Loss
```python
from gan.losses import HingeLoss

loss_fn = HingeLoss()
d_loss, g_loss = loss_fn(real_scores, fake_scores)
```

**Features:**
- Hinge loss for GANs
- Uses raw scores (before sigmoid)
- Good for high-quality generation

### 4. Feature Matching Loss
```python
from gan.losses import FeatureMatchingLoss

fm_loss = FeatureMatchingLoss(feature_weight=1.0)
loss = fm_loss(real_features, fake_features)
```

**Features:**
- Encourages similar features between real and fake images
- Improves training stability
- Can be combined with other losses

## ⚙️ Configuration

### Configuration Parameters

| Category | Parameter | Description | Default |
|----------|-----------|-------------|---------|
| **Model** | `latent_dim` | Dimension of latent space | 100 |
| | `hidden_dims` | Hidden dimensions for layers | [512, 256, 128, 64] |
| | `in_channels` | Number of input channels | 1 |
| | `image_size` | Size of input/output images | 28 |
| **Training** | `batch_size` | Training batch size | 64 |
| | `num_epochs` | Number of training epochs | 100 |
| | `g_learning_rate` | Generator learning rate | 2e-4 |
| | `d_learning_rate` | Discriminator learning rate | 2e-4 |
| | `weight_decay` | Weight decay for regularization | 1e-4 |
| | `max_grad_norm` | Maximum gradient norm | 1.0 |
| **Loss** | `loss_type` | Loss function type | 'gan' |
| | `label_smoothing` | Label smoothing factor | 0.1 |
| | `use_feature_matching` | Enable feature matching | False |
| | `feature_weight` | Feature matching weight | 1.0 |
| **Optimizer** | `optimizer_type` | Optimizer type | 'adam' |
| | `scheduler_type` | Learning rate scheduler | 'cosine' |
| | `warmup_steps` | Warmup steps for scheduler | 1000 |
| **EMA** | `ema_decay` | EMA decay rate | 0.9999 |

## Sampling Strategies

### 1. Random Sampling
Generate images by sampling from the prior distribution (standard normal).

### 2. Latent Grid
Create a 2D grid by varying two latent dimensions while keeping others fixed.

### 3. Latent Traversal
Vary one latent dimension while keeping all others fixed to understand what each dimension controls.

### 4. Latent Interpolation
Smoothly interpolate between two points in latent space to see continuous transitions.

## 📈 Training Features

### 1. Comprehensive Logging
- Training and validation losses
- Learning rate scheduling
- Model checkpoints
- Generated samples

### 2. EMA (Exponential Moving Average)
- Stabilizes training
- Improves sample quality
- Separate EMA for generator and discriminator

### 3. Gradient Clipping
- Prevents gradient explosion
- Configurable maximum gradient norm

### 4. Learning Rate Scheduling
- Cosine annealing
- Linear decay
- Step decay
- Warmup support

### 5. Checkpointing
- Regular checkpoints
- Best model saving
- Resume training capability

## 🐛 Troubleshooting

### Common Issues

1. **Mode Collapse**
   - Reduce learning rate
   - Use feature matching loss
   - Increase discriminator capacity
   - Try Wasserstein loss

2. **Training Instability**
   - Enable gradient clipping
   - Use label smoothing
   - Balance generator/discriminator updates
   - Check learning rates

3. **Poor Sample Quality**
   - Train for more epochs
   - Use EMA models
   - Increase model capacity
   - Try different loss functions

4. **Memory Issues**
   - Reduce batch size
   - Use smaller model
   - Enable gradient checkpointing
   - Use mixed precision training

## License

This implementation is part of the diffusion model project and follows the same license terms.


## 📖 References

1. **DCGAN Paper**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
2. **WGAN Paper**: "Wasserstein GAN"
3. **Hinge Loss**: "Improved Training of Wasserstein GANs"
4. **Feature Matching**: "Improved Techniques for Training GANs"
