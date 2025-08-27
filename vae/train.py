"""
VAE Trainer for training Variational Autoencoders.

This module provides a complete training pipeline for VAEs,
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
from vae.model import VAE
from vae.losses import VAELoss, BetaVAELoss
from diffusion.utils import (
    EMA, set_seed, gradient_clip, save_checkpoint, load_checkpoint, 
    get_device, log_hyperparameters, create_lr_scheduler, create_optimizer,
    create_dataloader, print_model_summary
)
from diffusion.train import plot_losses
import torchvision.utils as vutils


class VAETrainer:
    """
    VAE Trainer for MNIST dataset.
    Handles training loop, logging, and checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VAE trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        self.experiment_name = config['experiment_name']
        
        # Set seed for reproducibility
        set_seed(config['seed'])
        
        # Create VAE model
        self.model = VAE(
            in_channels=config['in_channels'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            image_size=config['image_size']
        )
        
        # Create loss function
        if config['loss_type'] == 'beta_vae':
            self.loss_fn = BetaVAELoss(
                beta_start=config['beta_start'],
                beta_end=config['beta_end'],
                beta_steps=config['beta_steps'],
                loss_type=config['reconstruction_loss_type']
            )
        else:
            self.loss_fn = VAELoss(
                beta=config['beta'],
                loss_type=config['reconstruction_loss_type']
            )
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            optimizer_type=config['optimizer_type'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_type=config['scheduler_type'],
            num_training_steps=config['num_epochs'] * config['steps_per_epoch'],
            warmup_steps=config['warmup_steps']
        )
        
        # Create EMA model
        self.ema = EMA(self.model, decay=config['ema_decay'])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
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
            # Normalize to [0, 1] for VAE (since we use sigmoid in decoder)
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
        images = batch[0].to(self.device)
        
        # Forward pass
        recon_images, mu, log_var = self.model(images)
        
        # Update beta if using beta-VAE
        if isinstance(self.loss_fn, BetaVAELoss):
            self.loss_fn.update_beta(self.global_step)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = self.loss_fn(recon_images, images, mu, log_var)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        gradient_clip(self.model, self.config['max_grad_norm'])
        
        # Update parameters
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Update EMA
        self.ema.update(self.model)
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta': self.loss_fn.beta if hasattr(self.loss_fn, 'beta') else self.config['beta']
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
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device)
                
                # Forward pass
                recon_images, mu, log_var = self.model(images)
                
                # Compute loss
                batch_total_loss, batch_recon_loss, batch_kl_loss = self.loss_fn(
                    recon_images, images, mu, log_var
                )
                
                total_loss += batch_total_loss.item()
                total_recon_loss += batch_recon_loss.item()
                total_kl_loss += batch_kl_loss.item()
                num_batches += 1
        
        self.model.train()
        
        return {
            'val_total_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }
    
    def save_checkpoint(self, is_best=False, normal_save=False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        if normal_save:
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'], 
                f"checkpoint_epoch_{self.current_epoch}.pth"
            )
        
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                epoch=self.current_epoch,
                loss=self.best_loss,
                ema=self.ema,
                filepath=checkpoint_path
            )
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], "best_model.pth")
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                epoch=self.current_epoch,
                loss=self.best_loss,
                ema=self.ema,
                filepath=best_path
            )
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            ema=self.ema,
            filepath=checkpoint_path
        )
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def generate_samples(self, num_samples=16, nrow=4):
        """
        Generate sample images from the VAE.
        
        Args:
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate samples
            samples = self.model.sample(num_samples=num_samples)
            
            # Save samples
            samples_path = os.path.join(
                self.config['sample_dir'], 
                f"samples_epoch_{self.current_epoch}.png"
            )
            
            # Convert to grid and save
            self._save_image_grid(samples, samples_path, nrow=nrow)
        
        self.model.train()
    
    def _save_image_grid(self, images, filepath, nrow=4):
        """
        Save images as a grid.
        
        Args:
            images: Tensor of images
            filepath: Path to save the grid
            nrow: Number of samples per row
        """
        # Create grid
        grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)
        
        # Save grid
        vutils.save_image(grid, filepath)
    
    def log_metrics(self, train_loss, val_loss, epoch):
        """
        Log training metrics.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            epoch: Current epoch
        """
        # Log to file
        log_file = os.path.join(self.config['log_dir'], "training_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n")
        
        # Print to console
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    def train(self):
        """
        Main training loop.
        """
        print(f"Starting VAE training on {self.device}")
        print(f"Experiment: {self.experiment_name}")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            self.model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                step_losses = self.train_step(batch)
                train_losses.append(step_losses)
                
                # Update progress bar
                pbar.set_postfix({
                    'Total Loss': f"{step_losses['total_loss']:.4f}",
                    'Recon Loss': f"{step_losses['recon_loss']:.4f}",
                    'KL Loss': f"{step_losses['kl_loss']:.4f}",
                    'Beta': f"{step_losses['beta']:.4f}"
                })
                
                self.global_step += 1
                
                # Generate samples periodically
                if self.global_step % self.config['sample_every'] == 0:
                    self.generate_samples()
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
            avg_train_recon = np.mean([l['recon_loss'] for l in train_losses])
            avg_train_kl = np.mean([l['kl_loss'] for l in train_losses])
            
            self.log_metrics(avg_train_loss, val_metrics['val_total_loss'], epoch + 1)
            
            # Save checkpoint if new best loss
            if val_metrics['val_total_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_total_loss']
                self.save_checkpoint(is_best=True)
            
            # Save checkpoint every N epochs
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
        'hidden_dims': [32, 64, 128],
        'latent_dim': 128,
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-4,
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
        'loss_type': 'vae', # 'vae' or 'beta_vae'
        'reconstruction_loss_type': 'mse', # 'mse' or 'bce'
        'beta': 1.0,
        'beta_start': 0.0,
        'beta_end': 1.0,
        'beta_steps': 1000,
        
        # Logging and saving
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        'data_dir': 'data',
        'results_dir': 'evaluation_results',
        'experiment_name': 'test_VAE',


    }

def load_config(config_path: str) -> dict:
    """
    Load configuration from a Python file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Get default config first
    config = get_default_config()
    
    # Load the config module
    import importlib
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Update with values from config file if they exist
    if hasattr(config_module, 'DEFAULT_CONFIG'):
        config.update(config_module.DEFAULT_CONFIG)
    
    return config

def main():
    """
    Main function to run VAE training.
    """
    parser = argparse.ArgumentParser(description="Train VAE on MNIST")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Override with command line arguments
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.resume:
        config['resume_from'] = args.resume

    # Create trainer and start training
    trainer = VAETrainer(config)
    trainer.train()

    # Plot losses
    plot_losses(os.path.join(trainer.config['log_dir'], "training_log.txt"), 
        os.path.join(trainer.config['log_dir'], "loss_plot.png"))


if __name__ == "__main__":
    main()
