"""
GAN Sampler for generating images from trained Generative Adversarial Networks.

This module provides functionality to load trained GAN models and
generate new images from the latent space.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional, Tuple, List
import pickle
import torchvision.utils as vutils

# Import our modules
from gan.model import GAN

# Import utility functions from diffusion module
from diffusion.utils import get_device


class GANSampler:
    """
    GAN Sampler for generating images from trained models.
    
    Provides various sampling strategies and visualization tools
    for exploring the latent space of trained GANs.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize GAN sampler.
        
        Args:
            model_path: Path to trained GAN model checkpoint
            device: Device to use for sampling
        """
        self.device = get_device()
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Loaded GAN model from {model_path}")
        print(f"Generator parameters: {sum(p.numel() for p in self.model.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.model.discriminator.parameters()):,}")
        print(f"Latent dimension: {self.model.latent_dim}")
        print(f"Hidden dimensions: {self.model.hidden_dims}")
        print(f"Input channels: {self.model.in_channels}")
        print(f"Image size: {self.model.image_size}")
        print(f"Using device: {self.device}")
    
    def _load_model(self, model_path: str) -> GAN:
        """
        Load GAN model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded GAN model
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration from checkpoint
        if 'model_state_dict' in checkpoint:
            # Load model state dict
            model_state = checkpoint['model_state_dict']
            
            # Extract model configuration from state dict
            config = self._extract_model_config(model_state)
            
            # Create model with extracted configuration
            model = GAN(
                latent_dim=config['latent_dim'],
                hidden_dims=config['hidden_dims'],
                in_channels=config['in_channels'],
                image_size=config['image_size']
            )
            
            # Load state dict
            model.load_state_dict(model_state)
            
        else:
            # Assume it's a direct model save
            model = checkpoint
        
        return model.to(self.device)
    
    def _extract_model_config(self, model_state: dict) -> dict:
        """
        Extract model configuration from state dict.
        
        Args:
            model_state: GAN state dictionary
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Extract latent dimension
        for key in model_state.keys():
            if 'generator.fc.weight' in key:
                config['latent_dim'] = model_state[key].shape[1]
                break
        
        if 'latent_dim' not in config:
            raise ValueError("Could not infer latent dimension from checkpoint")
        
        # Extract hidden dimensions from generator 
        # and input channels from discriminator
        hidden_dims = []
        for key in sorted(model_state.keys()):
            if 'generator.generator' in key and '.weight' in key and len(model_state[key].shape) == 4:
                # Extract layer number and output channels
                hidden_dims.append(model_state[key].shape[0])

            if 'discriminator.discriminator.0.weight' in key:  # First conv layer
                config['in_channels'] = model_state[key].shape[1]
        
        if not hidden_dims:
            # Default configuration if we can't extract
            hidden_dims = [1024, 512, 256, 128]
            print("Warning: Could not extract hidden dimensions, using default")
        
        config['hidden_dims'] = hidden_dims
        
        if 'in_channels' not in config:
            config['in_channels'] = 1  # Default for MNIST
        
        # Extract image size
        for key in model_state.keys():
            if 'generator.fc.weight' in key:
                initial_size = int(np.sqrt(model_state[key].shape[0] / (config['hidden_dims'][0])))
                config['image_size'] = initial_size * (2 ** len(config['hidden_dims']))
                break
        
        if 'image_size' not in config:
            config['image_size'] = 32
        
        print(f"Extracted model configuration:")
        print(f"  Latent dimension: {config['latent_dim']}")
        print(f"  Hidden dimensions: {config['hidden_dims']}")
        print(f"  Input channels: {config['in_channels']}")
        print(f"  Image size: {config['image_size']}")
        
        return config
    
    def sample_random(self, num_samples: int = 16) -> torch.Tensor:
        """
        Sample random images from the GAN.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated image tensor
        """
        with torch.no_grad():
            samples = self.model.sample(num_samples=num_samples)
        
        return samples
    
    def sample_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample images from specific latent vectors.
        
        Args:
            z: Latent vectors of shape (batch_size, latent_dim)
            
        Returns:
            Generated image tensor
        """
        z = z.to(self.device)
        with torch.no_grad():
            samples = self.model.generate(z)
        
        return samples
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, 
                          num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two latent vectors.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated images
        """
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        
        # Interpolate latent vectors
        interpolated_z = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated_z.append(z_interp)
        
        interpolated_z = torch.stack(interpolated_z)
        
        # Generate images
        with torch.no_grad():
            samples = self.model.generate(interpolated_z)
        
        return samples
    
    def traverse_latent_dimension(self, base_z: torch.Tensor, dim: int, 
                                 range_min: float = -3.0, range_max: float = 3.0,
                                 num_steps: int = 10) -> torch.Tensor:
        """
        Traverse a specific latent dimension while keeping others fixed.
        
        Args:
            base_z: Base latent vector
            dim: Dimension to traverse
            range_min: Minimum value for traversal
            range_max: Maximum value for traversal
            num_steps: Number of steps in traversal
            
        Returns:
            Traversed images
        """
        base_z = base_z.to(self.device)
        
        # Create traversal values
        values = torch.linspace(range_min, range_max, num_steps, device=self.device)
        
        # Create traversed latent vectors
        traversed_z = []
        for value in values:
            z_traverse = base_z.clone()
            z_traverse[0, dim] = value
            traversed_z.append(z_traverse)
        
        traversed_z = torch.cat(traversed_z, dim=0)
        
        # Generate images
        with torch.no_grad():
            samples = self.model.generate(traversed_z)
        
        return samples
    
    def generate_latent_grid(self, num_samples: int = 8, 
                           range_min: float = -3.0, range_max: float = 3.0) -> torch.Tensor:
        """
        Generate a grid of samples by varying two latent dimensions.
        
        Args:
            num_samples: Number of samples per dimension
            range_min: Minimum value for latent dimensions
            range_max: Maximum value for latent dimensions
            
        Returns:
            Grid of generated images
        """
        # Create grid of latent values
        values = torch.linspace(range_min, range_max, num_samples, device=self.device)
        
        # Create grid of latent vectors
        z_grid = []
        for i in range(num_samples):
            for j in range(num_samples):
                z = torch.zeros(1, self.model.latent_dim, device=self.device)
                z[0, 0] = values[i]  # First dimension
                z[0, 1] = values[j]  # Second dimension
                z_grid.append(z)
        
        z_grid = torch.cat(z_grid, dim=0)
        
        # Generate images
        with torch.no_grad():
            samples = self.model.generate(z_grid)
        
        return samples
    
    def save_samples(self, samples: torch.Tensor, filepath: str, 
                    nrow: int = 4):
        """
        Save generated samples as a grid image.
        
        Args:
            samples: Generated image tensor
            filepath: Path to save the image
            num_samples: Number of samples to display (if None, use all)
        """
        # Create grid
        grid = vutils.make_grid(samples, nrow=nrow, normalize=False, padding=2)
        
        grid_np = grid.cpu().numpy()
        if len(grid_np.shape) == 3 and grid_np.shape[0] == 3:
            # For grayscale images in RGB format, all channels are identical
            grid_np = grid_np[0]  # Take first channel

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_np, cmap='gray')
        plt.title(f"Samples")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save grid
        vutils.save_image(grid, filepath)
        
        print(f"Saved samples to {filepath}")
    
    def save_interpolation(self, interpolated_samples: torch.Tensor, 
                          filepath: str, title: str = "Latent Interpolation", nrow: int = 4):
        """
        Save interpolated samples as a sequence.
        
        Args:
            interpolated_samples: Interpolated image tensor
            filepath: Path to save the image
            title: Title for the plot
        """
        # Create grid
        grid = vutils.make_grid(interpolated_samples, nrow=nrow, normalize=False, padding=2)
        
        grid_np = grid.cpu().numpy()
        if len(grid_np.shape) == 3 and grid_np.shape[0] == 3:
            # For grayscale images in RGB format, all channels are identical
            grid_np = grid_np[0]  # Take first channel

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_np, cmap='gray')
        plt.title(f"{title}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save grid
        vutils.save_image(grid, filepath)
        
        print(f"Saved interpolation to {filepath}")
    
    def save_traversal(self, traversed_samples: torch.Tensor, 
                      filepath: str, dim: int, title: str = None, nrow: int = 4):
        """
        Save latent traversal as a sequence.
        
        Args:
            traversed_samples: Traversed image tensor
            filepath: Path to save the image
            dim: Dimension that was traversed
            title: Title for the plot
        """
        if title is None:
            title = f"Latent Dimension {dim} Traversal"
        grid = vutils.make_grid(traversed_samples, nrow=nrow, normalize=False, padding=2)
        
        grid_np = grid.cpu().numpy()
        if len(grid_np.shape) == 3 and grid_np.shape[0] == 3:
            # For grayscale images in RGB format, all channels are identical
            grid_np = grid_np[0]  # Take first channel

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_np, cmap='gray')
        plt.title(f"{title}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save grid
        vutils.save_image(grid, filepath)
        
        print(f"Saved traversal to {filepath}")
    
    def save_latent_grid(self, samples: torch.Tensor, filepath: str,
                        num_samples: int = 8, range_min: float = -3.0, range_max: float = 3.0):
        """
        Save latent grid as a 2D grid image.
        
        Args:
            samples: Generated grid samples
            filepath: Path to save the image
            num_samples: Number of samples per dimension
            range_min: Minimum value for latent dimensions
            range_max: Maximum value for latent dimensions
        """
        samples_np = (samples.cpu().numpy() + 1) / 2
        samples_np = np.clip(samples_np, 0, 1)
        
        # Reshape to grid
        samples_grid = samples_np.reshape(num_samples, num_samples, 1, 28, 28)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, num_samples, figsize=(12, 12))
        
        for i in range(num_samples):
            for j in range(num_samples):
                axes[i, j].imshow(samples_grid[i, j, 0], cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved latent grid to {filepath}")


def main():
    """
    Main function for GAN sampling.
    """
    parser = argparse.ArgumentParser(description="Sample from trained VAE")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained VAE model')
    parser.add_argument('--output_dir', type=str, default='vae_samples', help='Output directory')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=16, help='Number of random samples')

    # Interpolation parameters
    parser.add_argument('--interpolation', action='store_true', help='Generate interpolation')
    parser.add_argument('--interpolation_steps', type=int, default=10, help='Number of interpolation steps')
    
    # Latent traversal parameters
    parser.add_argument('--latent_traversal', action='store_true', help='Generate latent traversal')
    parser.add_argument('--traversal_steps', type=int, default=10, help='Number of traversal steps')
    parser.add_argument('--traverse_dim', type=int, default=0, help='Dimension to traverse')
    
    # Latent grid parameters
    parser.add_argument('--latent_grid', action='store_true', help='Generate latent grid')
    parser.add_argument('--latent_range_min', type=float, default=-3.0, help='Minimum latent value')
    parser.add_argument('--latent_range_max', type=float, default=3.0, help='Maximum latent value')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create sampler
    sampler = GANSampler(args.model_path)
    
    # Generate random samples
    print("Generating random samples...")
    random_samples = sampler.sample_random(args.num_samples)
    sampler.save_samples(random_samples, os.path.join(args.output_dir, "random_samples.png"))
    
    # Generate latent grid
    print("Generating latent grid...")
    if args.latent_grid:
        grid_samples = sampler.generate_latent_grid(
            num_samples=args.num_samples,
            range_min=args.latent_range_min,
            range_max=args.latent_range_max
        )   
        sampler.save_latent_grid(
            grid_samples, 
            os.path.join(args.output_dir, "latent_grid.png"),
            num_samples=args.num_samples,
            range_min=args.latent_range_min,
            range_max=args.latent_range_max
        )
    
    # Generate latent traversal
    print("Generating latent traversal...")
    if args.latent_traversal:
        base_z = torch.randn(1, sampler.model.latent_dim)
        traversed_samples = sampler.traverse_latent_dimension(
            base_z, 
            dim=args.traverse_dim,
            range_min=args.latent_range_min,
            range_max=args.latent_range_max,
            num_steps=args.traversal_steps
        )
        sampler.save_traversal(
            traversed_samples,
            os.path.join(args.output_dir, f"traversal_dim_{args.traverse_dim}.png"),
            dim=args.traverse_dim
        )
    
    # Generate interpolation between two random points
    print("Generating interpolation...")
    if args.interpolation:
        z1 = torch.randn(1, sampler.model.latent_dim)
        z2 = torch.randn(1, sampler.model.latent_dim)
        interpolated_samples = sampler.interpolate_latent(z1, z2, args.interpolation_steps)
        sampler.save_interpolation(
            interpolated_samples,
            os.path.join(args.output_dir, "interpolation.png")
        )
    
    print(f"All samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
