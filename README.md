# Generative Models for Image Generation

A collection of generative models for image generation, implemented in PyTorch and trained on the MNIST dataset.

## Overview

This project provides implementations of three generative model architectures:

- **Diffusion Models (DDPM)**: Denoising Diffusion Probabilistic Models with UNet architecture
- **Generative Adversarial Networks (GAN)**: DCGAN implementation with multiple loss functions
- **Variational Autoencoders (VAE)**: CNN-based VAE with disentanglement capabilities

Each model includes training pipelines, sampling utilities, and evaluation metrics.

## Project Structure

```
diffusion_model/
├── diffusion/ # DDPM implementation
│ ├── model.py          # UNet architecture and time embeddings
│ ├── schedulers.py     # Beta/alpha/sigma schedules, DDIM/EDM
│ ├── losses.py         # Epsilon vs x0 objectives
│ ├── sampler.py        # Ancestral & deterministic samplers
│ ├── conditioning.py   # Class labels, masks, text encoding
│ ├── train.py          # Training loops, EMA, logging
│ ├── sample.py         # Load checkpoint, generate grids
│ ├── utils.py          # EMA, gradient clip, seeding
│ ├── eval.py           # FID/IS/LPIPS/PSNR evaluation
│ └── README.md # Detailed documentation
├── gan/ # GAN implementation
│ ├── model.py # DCGAN architecture
│ ├── train.py # Training pipeline
│ ├── sample.py # Sampling utilities
│ └── README.md # Detailed documentation
├── vae/ # VAE implementation
│ ├── model.py # CNN-based VAE
│ ├── train.py # Training pipeline
│ ├── sample.py # Sampling utilities
│ └── README.md # Detailed documentation
├── data/ # Dataset storage
├── checkpoints/ # Model checkpoints
├── samples/ # Generated images
├── logs/ # Training logs
└── README.md # This overview
```

## Installation

```bash
# Clone the repository
git clone https://github.com/dj-shark31/ml-image-generators.git
cd diffusion_model

# Install dependencies
pip install -e .
```

## Usage

### Training Models

```bash
# Train Diffusion Model
python -m diffusion.train --num_epochs 100

# Train GAN
python -m gan.train --num_epochs 100

# Train VAE
python -m vae.train --num_epochs 100
```


### Generating Images

```bash
# Generate with Diffusion Model
python -m diffusion.sample --checkpoint checkpoints/diffusion/best_model.pth

# Generate with GAN
python gan.sample --model_path checkpoints/gan/best_model.pth

# Generate with VAE
python vae.sample --model_path checkpoints/vae/best_model.pth
```


### Evaluation

Evaluate model performance:

```bash
python -m diffusion.eval --checkpoint checkpoints/diffusion/best_model.pth --num-samples 1000
```


##  Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations
- **tqdm**: Progress bars
- **scipy**: Statistical computations

## License

This project is licensed under the MIT License.
