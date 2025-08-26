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

# Import our modules
from diffusion.model import UNetMini
from diffusion.schedulers import DDPMScheduler
from diffusion.losses import DDPMLoss
from diffusion.utils import EMA, set_seed, gradient_clip, save_checkpoint, load_checkpoint, get_device, log_hyperparameters, create_lr_scheduler


class Trainer:
    """
    DDPM Trainer for MNIST dataset.
    Handles training loop, logging, and checkpointing.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        self.experiment_name = config['experiment_name']
        
        # Set seed for reproducibility
        set_seed(config['seed'])
        
        # Create model, scheduler, and loss
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
        
        self.loss_fn = DDPMLoss(loss_type=config['loss_type'])
        
        # Create optimizer and learning rate scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            num_training_steps=config['num_epochs'] * config['steps_per_epoch'],
            scheduler_type=config['scheduler_type'],
            warmup_steps=config['warmup_steps']
        )
        
        # Create EMA model and move to device
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
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
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
        
        # Create dataloaders using utility function
        from diffusion.utils import create_dataloader
        
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
            Loss value
        """
        images = batch[0].to(self.device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.config['num_train_timesteps'], 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to images
        noisy_images, noise = self.scheduler.add_noise(images, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noisy_images, timesteps)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clip(self.model, self.config['max_grad_norm'])
        
        # Update parameters
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Update EMA
        self.ema.update(self.model)
        
        return loss.item()
    
    def validate(self, val_loader):
        """
        Validation step.
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device)
                batch_size = images.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.config['num_train_timesteps'], 
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise to images
                noisy_images, noise = self.scheduler.add_noise(images, timesteps)
                
                # Predict noise
                predicted_noise = self.model(noisy_images, timesteps)
                
                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
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
        print(f"Resumed from epoch {self.current_epoch}")
    
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
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            self.model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            for batch in train_pbar:
                loss = self.train_step(batch)
                train_losses.append(loss)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                self.global_step += 1
            
            # Validation phase
            val_loss = self.validate(val_loader)
            avg_train_loss = np.mean(train_losses)
            
            # Log metrics
            self.log_metrics(avg_train_loss, val_loss, epoch + 1)
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(normal_save=True)
            
            # Generate samples periodically
            if (epoch + 1) % self.config['sample_every'] == 0:
                self.generate_samples(epoch + 1)
        
        print("Training completed!")
    
    def generate_samples(self, epoch, num_samples=4):
        """
        Generate and save sample images.
        
        Args:
            epoch: Current epoch
            num_samples: Number of samples to generate
        """
        from diffusion.sampler import AncestralSampler
        
        # Use EMA model for generation
        self.ema.apply_shadow(self.model)
        self.model.eval()
        
        # Create sampler
        sampler = AncestralSampler(self.scheduler, num_inference_steps=1000)
        
        # Generate samples
        with torch.no_grad():
            samples = sampler(
                self.model, 
                (num_samples, 1, self.config['image_size'], self.config['image_size']),
                progress_bar=False
            )
        
        # Denormalize samples
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        samples = torch.clamp(samples, 0, 1)
        
        # Save samples
        sample_path = os.path.join(self.config['sample_dir'], f"samples_epoch_{epoch}.png")
        self.save_image_grid(samples, sample_path, nrow=4)
        
        # Restore original model
        self.ema.restore(self.model)
        self.model.train()
        
        print(f"Samples saved to {sample_path}")
    
    def save_image_grid(self, images, filepath, nrow=4):
        """
        Save a grid of images.
        
        Args:
            images: Tensor of images
            filepath: Path to save the grid
            nrow: Number of images per row
        """
        import torchvision.utils as vutils
        
        # Create grid
        grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)
        
        # Save grid
        vutils.save_image(grid, filepath)


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
        'out_channels': 1,
        'model_channels': 64,
        'time_emb_dim': 128,
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'scheduler_type': 'cosine',
        'warmup_steps': 1000,
        'weight_decay': 1e-4,
        'num_workers': 4,
        
        # Diffusion parameters
        'num_train_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        
        # Loss parameters
        'loss_type': 'l2',
        
        # Optimization parameters
        'max_grad_norm': 1.0,
        'ema_decay': 0.9999,
        
        # Logging and saving
        'save_every': 10,
        'sample_every': 5,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        'data_dir': 'data',
        
        # Misc
        'seed': 42,
        'resume_from': None,
        'steps_per_epoch': 469,  # MNIST train size / batch_size
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
    if hasattr(config_module, 'MODEL_CONFIG'):
        config.update(config_module.MODEL_CONFIG)
    if hasattr(config_module, 'TRAINING_CONFIG'):
        config.update(config_module.TRAINING_CONFIG)
    if hasattr(config_module, 'DIFFUSION_CONFIG'):
        config.update(config_module.DIFFUSION_CONFIG)
    if hasattr(config_module, 'PATHS_CONFIG'):
        config.update(config_module.PATHS_CONFIG)
    if hasattr(config_module, 'SAMPLING_CONFIG'):
        config.update(config_module.SAMPLING_CONFIG)
    if hasattr(config_module, 'EVALUATION_CONFIG'):
        config.update(config_module.EVALUATION_CONFIG)
    
    return config

def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')
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
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 