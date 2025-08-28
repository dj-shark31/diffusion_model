"""
Generative Adversarial Network (GAN) model with DCGAN architecture.

This module implements a GAN using DCGAN architecture with convolutional
neural networks for both generator and discriminator, specifically designed for MNIST images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# Import utility functions from diffusion module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.utils import get_device, count_parameters


class Generator(nn.Module):
    """
    DCGAN Generator.
    
    Generates images from random noise using transposed convolutions.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dims: list = [512, 256, 128, 64],
                 out_channels: int = 1,
                 image_size: int = 32):
        """
        Initialize the generator.
        
        Args:
            latent_dim: Dimension of the latent space (noise input)
            hidden_dims: List of hidden dimensions for each layer
            out_channels: Number of output channels (1 for MNIST)
            image_size: Size of output images (assumed square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.image_size = image_size
        
        # Calculate the initial size after convolutions
        self.initial_size = image_size // (2 ** len(hidden_dims))
        
        # Initial linear layer to project from latent to conv input
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.initial_size ** 2)
        
        # Build generator layers
        modules = []
        in_channels = hidden_dims[0]
        
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(in_channels, hidden_dims[i + 1], 
                                  kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dims[i + 1]),
                nn.SiLU()
            ])
            in_channels = hidden_dims[i + 1]
        
        # Final layer to output channels
        modules.extend([
            nn.ConvTranspose2d(in_channels, self.out_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        ])
        
        self.generator = nn.Sequential(*modules)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated image tensor of shape (batch_size, channels, height, width)
        """
        # Project from latent to conv input
        x = self.fc(z)
        
        # Reshape to conv input
        x = x.view(x.size(0), self.hidden_dims[0], self.initial_size, self.initial_size)
        
        # Generate through transposed convolutions
        x = self.generator(x)
        
        return x


class Discriminator(nn.Module):
    """
    DCGAN Discriminator.
    
    Classifies images as real or fake using convolutional layers.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 hidden_dims: list = [64, 128, 256, 512],
                 image_size: int = 32):
        """
        Initialize the discriminator.
        
        Args:
            in_channels: Number of input channels (1 for MNIST)
            hidden_dims: List of hidden dimensions for each layer
            image_size: Size of input images (assumed square)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.image_size = image_size
        
        # Build discriminator layers
        modules = []
        in_channels = self.in_channels
        
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_channels = h_dim
        
        self.discriminator = nn.Sequential(*modules)
        
        # Calculate the size after convolutions
        self.conv_output_size = image_size // (2 ** len(hidden_dims))
        self.conv_output_channels = hidden_dims[-1]
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_channels * self.conv_output_size ** 2, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        # Discriminate through convolutional layers
        x = self.discriminator(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features from the discriminator.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        x = self.discriminator(x)
        x = x.view(x.size(0), -1)
        return x


class GAN(nn.Module):
    """
    Generative Adversarial Network (GAN) with DCGAN architecture.
    
    Combines generator and discriminator for training and sampling.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dims: list = [512, 256, 128, 64],
                 in_channels: int = 1,
                 image_size: int = 32):
        """
        Initialize the GAN.
        
        Args:
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden dimensions for generator/discriminator
            in_channels: Number of input channels (1 for MNIST)
            image_size: Size of input/output images (assumed square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.image_size = image_size
        
        # Create generator and discriminator
        self.generator = Generator(latent_dim, hidden_dims, in_channels, image_size)
        self.discriminator = Discriminator(in_channels, list(reversed(hidden_dims)), image_size)
        
        # Move to device
        self.device = get_device()
        self.to(self.device)
        
    
    def sample(self, num_samples: int = 1, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample images from the GAN.
        
        Args:
            num_samples: Number of samples to generate
            z: Optional latent vectors to generate from (if None, sample from prior)
            
        Returns:
            Generated image tensor
        """
        if z is None:
            # Sample from prior (uniform distribution)
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # Generate
        with torch.no_grad():
            samples = self.generator(z)
        
        return samples


if __name__ == "__main__":
    # Test the GAN model
    device = get_device()
    
    # Create GAN
    model = GAN(
        latent_dim=128,
        hidden_dims=[512, 256, 128, 64],
        in_channels=1,
        image_size=32
    )
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, model.latent_dim).to(device)
    real_images = torch.randn(batch_size, 1, 32, 32).to(device)
    
    # Generate fake images
    fake_images = model.generator(z)
    
    # Discriminate
    real_probs = model.discriminator(real_images)
    fake_probs = model.discriminator(fake_images)
    
    print(f"GAN Model Summary:")
    print(f"  Generator parameters: {count_parameters(model.generator):,}")
    print(f"  Discriminator parameters: {count_parameters(model.discriminator):,}")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Input shape: {z.shape}")
    print(f"  Generated shape: {fake_images.shape}")
    print(f"  Real probs shape: {real_probs.shape}")
    print(f"  Fake probs shape: {fake_probs.shape}")
    print(f"  Using device: {device}")
    
    # Test sampling
    samples = model.sample(num_samples=4)
    print(f"  Sampled shape: {samples.shape}")
