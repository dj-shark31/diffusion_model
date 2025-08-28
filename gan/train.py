"""
GAN Trainer for training Generative Adversarial Networks.

This module provides a complete training pipeline for GANs,
reusing functions from the diffusion module for consistency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

# Import our modules
from gan.model import GAN
from gan.losses import GANLoss, WassersteinLoss, HingeLoss, FeatureMatchingLoss


from diffusion.utils import (
    EMA, set_seed, gradient_clip, save_checkpoint, load_checkpoint, 
    get_device, log_hyperparameters, create_lr_scheduler, create_optimizer,
    create_dataloader, log_metrics, plot_losses, load_config, print_model_summary
)


class GANTrainer:
    """
    GAN Trainer for MNIST dataset.
    Handles training loop, logging, and checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GAN trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        self.experiment_name = config['experiment_name']
        
        # Set seed for reproducibility
        set_seed(config['seed'])
        
        # Create GAN model
        self.model = GAN(
            latent_dim=config['latent_dim'],
            hidden_dims=config['hidden_dims'],
            in_channels=config['in_channels'],
            image_size=config['image_size']
        )
        
        # Create loss function
        if config['loss_type'] == 'wasserstein':
            self.loss_fn = WassersteinLoss()
        elif config['loss_type'] == 'hinge':
            self.loss_fn = HingeLoss()
        else:
            self.loss_fn = GANLoss(label_smoothing=config['label_smoothing'])
        
        # Create feature matching loss if enabled
        if config.get('use_feature_matching', False):
            self.feature_loss = FeatureMatchingLoss(feature_weight=config.get('feature_weight', 1.0))
        else:
            self.feature_loss = None
        
        # Create optimizers
        self.g_optimizer = create_optimizer(
            self.model.generator,
            optimizer_type=config['optimizer_type'],
            learning_rate=config['g_learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.d_optimizer = create_optimizer(
            self.model.discriminator,
            optimizer_type=config['optimizer_type'],
            learning_rate=config['d_learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate schedulers
        self.g_scheduler = create_lr_scheduler(
            optimizer=self.g_optimizer,
            scheduler_type=config['scheduler_type'],
            num_training_steps=config['num_epochs'] * config['steps_per_epoch'],
            warmup_steps=config['warmup_steps']
        )
        
        self.d_scheduler = create_lr_scheduler(
            optimizer=self.d_optimizer,
            scheduler_type=config['scheduler_type'],
            num_training_steps=config['num_epochs'] * config['steps_per_epoch'],
            warmup_steps=config['warmup_steps']
        )
        
        # Create EMA models
        self.g_ema = EMA(self.model.generator, decay=config['ema_decay'])
        self.d_ema = EMA(self.model.discriminator, decay=config['ema_decay'])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['sample_dir'], exist_ok=True)
        config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], self.experiment_name)
        config['log_dir'] = os.path.join(config['log_dir'], self.experiment_name)
        config['sample_dir'] = os.path.join(config['sample_dir'], self.experiment_name)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['sample_dir'], exist_ok=True)
        
        # Log hyperparameters
        log_hyperparameters(config, config['log_dir'])

        # Print model summary
        print_model_summary(self.model)

        # Load checkpoint if exists
        if config['resume_from']:
            self.load_checkpoint(config['resume_from'])
    
    def create_dataloaders(self):
        """
        Create training and validation dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Data transformations
        transform = transforms.Compose([
            transforms.Resize(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for Tanh output
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(
            root=self.config['data_dir'],
            train=True,
            download=True,
            transform=transform
        )
        
        val_dataset = datasets.MNIST(
            root=self.config['data_dir'],
            train=False,
            download=True,
            transform=transform
        )
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            device=self.device
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            device=self.device
        )
        
        return train_loader, val_loader
    
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: Batch of images
            
        Returns:
            Dictionary containing loss values
        """
        real_images = batch[0].to(self.device)
        batch_size = real_images.size(0)
        
        # Generate noise
        z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake images
        fake_images = self.model.generator(z)
        
        # Get discriminator outputs
        if isinstance(self.loss_fn, (WassersteinLoss, HingeLoss)):
            # For Wasserstein and Hinge loss, use raw scores (before sigmoid)
            real_scores = self.model.discriminator.discriminator(real_images)
            fake_scores = self.model.discriminator.discriminator(fake_images.detach())
            
            # Flatten and get final scores
            real_scores = real_scores.view(real_scores.size(0), -1)
            fake_scores = fake_scores.view(fake_scores.size(0), -1)
            real_scores = self.model.discriminator.classifier(real_scores)
            fake_scores = self.model.discriminator.classifier(fake_scores)
            
            d_loss, _ = self.loss_fn(real_scores, fake_scores)
        else:
            # For standard GAN loss, use probabilities
            real_probs = self.model.discriminator(real_images)
            fake_probs = self.model.discriminator(fake_images.detach())
            d_loss, _ = self.loss_fn(real_probs, fake_probs)
        
        d_loss.backward()

        gradient_clip(self.model.discriminator, self.config['max_grad_norm'])

        self.d_optimizer.step()

        self.d_scheduler.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        # Generate new fake images
        fake_images = self.model.generator(z)
        
        # Get discriminator outputs for generator training
        if isinstance(self.loss_fn, (WassersteinLoss, HingeLoss)):
            fake_scores = self.model.discriminator.discriminator(fake_images)
            fake_scores = fake_scores.view(fake_scores.size(0), -1)
            fake_scores = self.model.discriminator.classifier(fake_scores)
            _, g_loss = self.loss_fn(None, fake_scores)
        else:
            fake_probs = self.model.discriminator(fake_images)
            _, g_loss = self.loss_fn(None, fake_probs)
        
        # Add feature matching loss if enabled
        if self.feature_loss is not None:
            real_features = self.model.discriminator.get_features(real_images)
            fake_features = self.model.discriminator.get_features(fake_images)
            fm_loss = self.feature_loss(real_features, fake_features)
            g_loss = g_loss + fm_loss
        else:
            fm_loss = torch.tensor(0.0, device=self.device)
        
        g_loss.backward()

        gradient_clip(self.model.generator, self.config['max_grad_norm'])
        
        self.g_optimizer.step()

        self.g_scheduler.step()
        
        # Update EMA
        self.g_ema.update(self.model.generator)
        self.d_ema.update(self.model.discriminator)
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'fm_loss': fm_loss.item() if self.feature_loss is not None else 0.0
        }
    
    def validate(self, val_loader):
        """
        Validation step.
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                real_images = batch[0].to(self.device)
                batch_size = real_images.size(0)
                
                # Generate noise
                z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
                
                # Generate fake images
                fake_images = self.model.generator(z)
                
                # Compute losses
                if isinstance(self.loss_fn, (WassersteinLoss, HingeLoss)):
                    real_scores = self.model.discriminator.discriminator(real_images)
                    fake_scores = self.model.discriminator.discriminator(fake_images)
                    real_scores = real_scores.view(real_scores.size(0), -1)
                    fake_scores = fake_scores.view(fake_scores.size(0), -1)
                    real_scores = self.model.discriminator.classifier(real_scores)
                    fake_scores = self.model.discriminator.classifier(fake_scores)
                    
                    d_loss, g_loss = self.loss_fn(real_scores, fake_scores)
                else:
                    real_probs = self.model.discriminator(real_images)
                    fake_probs = self.model.discriminator(fake_images)
                    d_loss, g_loss = self.loss_fn(real_probs, fake_probs)
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                num_batches += 1
        
        self.model.train()
        
        return {
            'val_d_loss': total_d_loss / num_batches,
            'val_g_loss': total_g_loss / num_batches
        }
    
    def save_checkpoint(self, is_best=False, normal_save=False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'g_loss': self.best_g_loss,
            'd_loss': self.best_d_loss,
            'g_ema_shadow': self.g_ema.shadow,
            'd_ema_shadow': self.d_ema.shadow
        }
        
        if normal_save:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], f"checkpoint_epoch_{self.current_epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        
        if 'g_ema_shadow' in checkpoint:
            self.g_ema.shadow = checkpoint['g_ema_shadow']
        if 'd_ema_shadow' in checkpoint:
            self.d_ema.shadow = checkpoint['d_ema_shadow']
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_g_loss = checkpoint['g_loss']
        self.best_d_loss = checkpoint['d_loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def generate_samples(self, num_samples=16):
        """
        Generate sample images from the GAN.
        
        Args:
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate samples
            samples = self.model.sample(num_samples=num_samples)
            
            # Save samples
            samples_path = os.path.join(self.config['sample_dir'], f"samples_epoch_{self.current_epoch}.png")
            
            # Convert to grid and save
            self._save_image_grid(samples, samples_path, num_samples)
        
        self.model.train()
    
    def _save_image_grid(self, images, filepath, num_samples):
        """
        Save images as a grid.
        
        Args:
            images: Tensor of images
            filepath: Path to save the grid
            num_samples: Number of samples
        """
        # Convert to numpy and denormalize from [-1, 1] to [0, 1]
        images = (images.cpu().numpy() + 1) / 2
        images = np.clip(images, 0, 1)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < num_samples:
                    axes[i, j].imshow(images[idx, 0], cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """
        Main training loop.
        """
        print(f"Starting GAN training on {self.device}")
        print(f"Experiment: {self.experiment_name}")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                step_losses = self.train_step(batch)
                train_losses.append(step_losses)
                
                # Update progress bar
                pbar.set_postfix({
                    'D Loss': f"{step_losses['d_loss']:.4f}",
                    'G Loss': f"{step_losses['g_loss']:.4f}",
                    'FM Loss': f"{step_losses['fm_loss']:.4f}"
                })
                
                self.global_step += 1
                
                # Generate samples periodically
                if self.global_step % self.config['sample_every'] == 0:
                    self.generate_samples()
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            avg_d_loss = np.mean([l['d_loss'] for l in train_losses])
            avg_g_loss = np.mean([l['g_loss'] for l in train_losses])
            avg_fm_loss = np.mean([l['fm_loss'] for l in train_losses])
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}, FM: {avg_fm_loss:.4f}")
            print(f"  Val   - D: {val_metrics['val_d_loss']:.4f}, G: {val_metrics['val_g_loss']:.4f}")

            log_metrics(self.config['log_dir'], avg_d_loss, val_metrics['val_d_loss'], epoch + 1)
            
            # Save checkpoint
            if val_metrics['val_g_loss'] < self.best_g_loss:
                self.best_g_loss = val_metrics['val_g_loss']
                self.best_d_loss = val_metrics['val_d_loss']
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(normal_save=True)
            
            # Generate samples at end of epoch
            if (epoch + 1) % self.config['sample_every'] == 0:
                self.generate_samples()
        
        print("Training completed!")


def get_default_config():
    """
    Get default configuration for training.
    
    Returns:
        Configuration dictionary
    """
    return {
        # Model parameters
        'image_size': 32,
        'in_channels': 1,
        'hidden_dims': [256, 128, 64],
        'latent_dim': 128,
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 128,
        'g_learning_rate': 1e-4,
        'd_learning_rate': 1e-4,
        'scheduler_type': 'cosine', # 'cosine' or 'linear' or 'constant'
        'warmup_steps': 1000,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'optimizer_type': 'adamw', # 'adamw' or 'adam' or 'sgd'
        'max_grad_norm': 1.0,
        'ema_decay': 0.9999,
        'save_every': 10,
        'sample_every': 5,
        'seed': 42,
        'resume_from': None,
        'steps_per_epoch': 469,  # MNIST train size / batch_size

        # Loss parameters
        'loss_type': 'gan', # 'gan' or 'wasserstein' or 'hinge'
        'label_smoothing': 0.1,
        'use_feature_matching': False,
        'feature_weight': 1.0,
        
        # Logging and saving
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        'data_dir': 'data',
        'results_dir': 'evaluation_results',
        'experiment_name': 'test_GAN',


    }

def main():
    """
    Main function to run GAN training.
    """
    parser = argparse.ArgumentParser(description="Train VAE on MNIST")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_default_config()
    if args.config:
        config = load_config(args.config, config)


    # Override with command line arguments
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['g_learning_rate'] = args.lr
        config['d_learning_rate'] = args.lr
    if args.resume:
        config['resume_from'] = args.resume
    
    # Create trainer and start training
    trainer = GANTrainer(config)
    trainer.train()

    # Plot losses
    plot_losses(os.path.join(trainer.config['log_dir'], "training_log.txt"), 
        os.path.join(trainer.config['log_dir'], "loss_plot.png"))


if __name__ == "__main__":
    main()
