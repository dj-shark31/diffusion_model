import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Import our modules
from diffusion.model import UNetMini
from diffusion.schedulers import DDPMScheduler
from diffusion.sampler import AncestralSampler, DeterministicSampler, DDIMSampler, create_sampler
from diffusion.utils import get_device, set_seed, load_checkpoint


class Sampler:
    """
    DDPM Sampler for generating images from trained models.
    """
    
    def __init__(self, config):
        """
        Initialize the sampler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(config['seed'])
        
        # Create model and scheduler
        self.model = UNetMini(
            image_size=config['image_size'],
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            model_channels=config['model_channels'],
            time_emb_dim=config['time_emb_dim']
        )
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config['num_train_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            beta_schedule=config['beta_schedule']
        )
        
        # Load checkpoint
        if config['checkpoint_path']:
            self.load_checkpoint(config['checkpoint_path'])
        else:
            print("No checkpoint provided, using random weights")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA if available
        if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
            print("Using EMA model for generation")
            # Apply EMA weights
            for name, param in self.model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    param.data = checkpoint['ema_shadow'][name]
        
        self.model.eval()
        print(f"Checkpoint loaded successfully (Epoch: {checkpoint.get('epoch', 'Unknown')})")
    
    def generate_samples(self, num_samples=4, sampler_type="ancestral", 
                        num_inference_steps=1000, eta=1.0, progress_bar=True):
        """
        Generate samples using the trained model.
        
        Args:
            num_samples: Number of samples to generate
            sampler_type: Type of sampler ("ancestral", "deterministic", "ddim")
            num_inference_steps: Number of inference steps
            eta: Noise level for DDIM (0 = deterministic, 1 = stochastic)
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        print(f"Generating {num_samples} samples using {sampler_type} sampler...")
        
        # Create sampler
        if sampler_type == "deterministic":
            sampler = create_sampler(
                sampler_type=sampler_type,
                scheduler=self.scheduler,
                num_inference_steps=num_inference_steps
            )
        else:
            sampler = create_sampler(
                sampler_type=sampler_type,
                scheduler=self.scheduler,
                num_inference_steps=num_inference_steps,
                eta=eta
            )

        
        # Generate samples
        with torch.no_grad():
            samples = sampler(
                self.model,
                (num_samples, 1, self.config['image_size'], self.config['image_size']),
                progress_bar=progress_bar
            )

        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def save_samples(self, samples, filepath, nrow=4):
        """
        Save samples as a grid image.
        
        Args:
            samples: Generated samples
            filepath: Path to save the image
            nrow: Number of images per row
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
        print(f"Samples saved to {filepath}")
    
    def generate_and_save(self, num_samples=4, sampler_type="ancestral", 
                         num_inference_steps=1000, eta=1.0, output_dir="samples"):
        """
        Generate samples and save them.
        
        Args:
            num_samples: Number of samples to generate
            sampler_type: Type of sampler
            num_inference_steps: Number of inference steps
            eta: Noise level for DDIM
            output_dir: Directory to save samples
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate samples
        samples = self.generate_samples(
            num_samples=num_samples,
            sampler_type=sampler_type,
            num_inference_steps=num_inference_steps,
            eta=eta
        )
        
        # Save samples
        filename = f"samples_{sampler_type}_steps{num_inference_steps}.png"
        if eta > 0:
            filename = f"samples_{sampler_type}_steps{num_inference_steps}_eta{eta}.png"
        
        filepath = os.path.join(output_dir, filename)
        self.save_samples(samples, filepath)
        
        return samples
    
    def generate_comparison(self, num_samples=16, output_dir="samples"):
        """
        Generate samples using different samplers for comparison.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save samples
        """
        print("Generating comparison samples...")
        
        samplers = [
            ("deterministic", 1000, 0.0),
            ("ddim", 50, 0.0),
            ("ddim", 50, 0.5),
            ("ancestral", 1000, 1.0)
        ]
        
        all_samples = []
        sampler_names = []
        
        for sampler_type, steps, eta in samplers:
            print(f"Generating with {sampler_type} (steps={steps}, eta={eta})")
            samples = self.generate_samples(
                num_samples=num_samples,
                sampler_type=sampler_type,
                num_inference_steps=steps,
                eta=eta,
                progress_bar=False
            )
            all_samples.append(samples)
            sampler_names.append(f"{sampler_type}_steps{steps}_eta{eta}")
        
        # Create comparison grid
        comparison_samples = torch.cat(all_samples, dim=0)
        
        # Save comparison
        filepath = os.path.join(output_dir, "sampler_comparison.png")
        self.save_samples(comparison_samples, filepath, nrow=num_samples)
        
        # Save individual samplers
        for i, (samples, name) in enumerate(zip(all_samples, sampler_names)):
            filepath = os.path.join(output_dir, f"comparison_{name}.png")
            self.save_samples(samples, filepath, nrow=4)
        
        print("Comparison samples saved!")
    
    def generate_progressive(self, num_samples=4, output_dir="samples", eta=0.0):
        """
        Generate progressive samples showing the denoising process.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save samples
        """
        print("Generating progressive samples...")
        
        # Start from pure noise
        samples = torch.randn(
            num_samples, 1, self.config['image_size'], self.config['image_size'],
            device=self.device
        )
        
        # Save initial noise
        os.makedirs(output_dir, exist_ok=True)
        initial_path = os.path.join(output_dir, "progressive_noise.png")
        self.save_samples(samples, initial_path, nrow=num_samples)
        
        # Progressive denoising
        self.scheduler.set_timesteps(num_inference_steps=1000)
        timesteps = self.scheduler.timesteps
        
        for i, timestep in enumerate(tqdm(timesteps, desc="Progressive denoising")):
            with torch.no_grad():
                # Predict noise
                predicted_noise = self.model(samples, timestep)
                
                # Denoise step
                samples = self.scheduler.step(predicted_noise, timestep, samples, eta=eta)
            
            # Save every 10 steps
            if (i + 1) % 10 == 0:
                step_path = os.path.join(output_dir, f"progressive_step_{i+1:03d}.png")
                self.save_samples(samples, step_path, nrow=num_samples)
        
        # Save final result
        final_path = os.path.join(output_dir, "progressive_final.png")
        self.save_samples(samples, final_path, nrow=num_samples)
        
        print("Progressive samples saved!")


def get_default_config():
    """
    Get default configuration for sampling.
    
    Returns:
        Configuration dictionary
    """
    return {
        # Model parameters
        'image_size': 32,
        'in_channels': 1,
        'out_channels': 1,
        'model_channels': 64,
        'time_emb_dim': 128,
        
        # Diffusion parameters
        'num_train_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        
        # Evaluation parameters
        'checkpoint_path': None,
        'batch_size': 32,
        'num_workers': 4,
        'data_dir': 'data',
        'results_dir': 'evaluation_results',
        'seed': 42,
    }

def load_config(config_path):
    """
    Load configuration from a file.
    """
    config = {}
    default_config = get_default_config()
    with open(config_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            # Try to convert value to int or float if possible
            try:
                if '.' in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                config[key] = value.strip()
    for key, value in default_config.items():
        if key not in config:
            print(f"Warning: {key} not found in config file, using default value: {value}")
            config[key] = value
    return config

def main():
    """
    Main sampling function.
    """
    parser = argparse.ArgumentParser(description='Generate samples from trained DDPM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--sampler', type=str, default='deterministic', 
                       choices=['ancestral', 'deterministic', 'ddim'], help='Sampler type')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--eta', type=float, default=1.0, help='Noise level for DDIM')
    parser.add_argument('--output-dir', type=str, default='samples', help='Output directory')
    parser.add_argument('--comparison', action='store_true', help='Generate comparison samples')
    parser.add_argument('--progressive', action='store_true', help='Generate progressive samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config: 
        config = load_config(args.config)
    else:
        config = get_default_config()

    config['checkpoint_path'] = args.checkpoint
    config['seed'] = args.seed
    
    # Create sampler
    sampler = Sampler(config)
    
    # Generate samples based on arguments
    if args.comparison:
        sampler.generate_comparison(num_samples=args.num_samples, output_dir=args.output_dir)
    elif args.progressive:
        sampler.generate_progressive(num_samples=args.num_samples, output_dir=args.output_dir, eta=args.eta)
    else:
        sampler.generate_and_save(
            num_samples=args.num_samples,
            sampler_type=args.sampler,
            num_inference_steps=args.steps,
            eta=args.eta,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main() 