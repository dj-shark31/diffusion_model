"""
Variational Autoencoder (VAE) model with CNN architecture.

This module implements a VAE using convolutional neural networks for
downsampling and upsampling, specifically designed for MNIST images.
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


class Encoder(nn.Module):
    """
    CNN Encoder for VAE.
    
    Downsamples input images and outputs mean and log variance
    for the latent space distribution.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 hidden_dims: list = [32, 64, 128],
                 latent_dim: int = 128,
                 image_size: int = 32):
        """
        Initialize the encoder.
        
        Args:
            in_channels: Number of input channels (1 for MNIST)
            hidden_dims: List of hidden dimensions for each layer
            latent_dim: Dimension of the latent space
            image_size: Size of input images (assumed square)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate the size after convolutions
        self.conv_output_size = image_size // (2 ** len(hidden_dims))
        self.conv_output_channels = hidden_dims[-1]
        
        # Build encoder layers
        modules = []
        in_channels = self.in_channels
        
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.SiLU()
            ])
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the flattened size for the linear layers
        self.flatten_size = self.conv_output_channels * self.conv_output_size ** 2
        
        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (mu, log_var) for the latent distribution
        """
        # Encode through convolutional layers
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    CNN Decoder for VAE.
    
    Upsamples from latent space to reconstruct images.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dims: list = [128, 64, 32],
                 out_channels: int = 1,
                 image_size: int = 32):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden dimensions for each layer (reverse of encoder)
            out_channels: Number of output channels (1 for MNIST)
            image_size: Size of output images (assumed square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.image_size = image_size
        
        # Calculate the size after convolutions
        self.conv_input_size = image_size // (2 ** len(hidden_dims))
        self.conv_input_channels = hidden_dims[0]
        
        # Linear layer to project from latent to conv input
        self.fc = nn.Linear(latent_dim, self.conv_input_channels * self.conv_input_size ** 2)
        
        # Build decoder layers
        modules = []
        in_channels = self.conv_input_channels
        
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(in_channels, hidden_dims[i + 1], 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, hidden_dims[i + 1]),
                nn.SiLU()
            ])
            in_channels = hidden_dims[i + 1]
        
        # Final layer to output channels
        modules.extend([
            nn.ConvTranspose2d(in_channels, self.out_channels, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        ])
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed image tensor of shape (batch_size, channels, height, width)
        """
        # Project from latent to conv input
        x = self.fc(z)
        
        # Reshape to conv input
        x = x.view(x.size(0), self.conv_input_channels, self.conv_input_size, self.conv_input_size)
        
        # Decode through transposed convolutions
        x = self.decoder(x)
        
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with CNN architecture.
    
    Combines encoder and decoder with reparameterization trick
    for training and sampling.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 hidden_dims: list = [32, 64, 128],
                 latent_dim: int = 128,
                 image_size: int = 32):
        """
        Initialize the VAE.
        
        Args:
            in_channels: Number of input channels (1 for MNIST)
            hidden_dims: List of hidden dimensions for encoder/decoder
            latent_dim: Dimension of the latent space
            image_size: Size of input/output images (assumed square)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Create encoder and decoder
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim, image_size)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), in_channels, image_size)
        
        # Move to device
        self.device = get_device()
        self.to(self.device)
        
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vectors
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (reconstructed_x, mu, log_var)
        """
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var
    
    def sample(self, num_samples: int = 1, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample images from the VAE.
        
        Args:
            num_samples: Number of samples to generate
            z: Optional latent vectors to decode (if None, sample from prior)
            
        Returns:
            Generated image tensor
        """
        if z is None:
            # Sample from prior (standard normal)
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # Decode
        with torch.no_grad():
            samples = self.decoder(z)
        
        return samples
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without sampling (deterministic).
        
        Args:
            x: Input tensor
            
        Returns:
            Mean of latent distribution
        """
        mu, _ = self.encoder(x)
        return mu


if __name__ == "__main__":
    # Test the VAE model
    device = get_device()
    
    # Create VAE
    vae = VAE(
        in_channels=1,
        hidden_dims=[32, 64, 128],
        latent_dim=128,
        image_size=32
    )
    
    # Test forward pass
    x = torch.randn(4, 1, 32, 32).to(device)
    recon_x, mu, log_var = vae(x)
    
    print(f"VAE Model Summary:")
    print(f"  Total parameters: {count_parameters(vae):,}")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstructed shape: {recon_x.shape}")
    print(f"  Latent shape: {mu.shape}")
    print(f"  Using device: {device}")
    
    # Test sampling
    samples = vae.sample(num_samples=4)
    print(f"  Sampled shape: {samples.shape}")
