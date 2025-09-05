import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Dict, Any, Optional
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import importlib
import torchvision.utils as vutils

class EMA:
    """
    Exponential Moving Average for model parameters.
    Helps stabilize training and improve model performance.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters on the same device as the model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
    
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        
        Args:
            model: Current model
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
    
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply EMA parameters to model.
        
        Args:
            model: Model to apply EMA to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """
        Restore original model parameters.
        
        Args:
            model: Model to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def gradient_clip(model: nn.Module, max_norm: float = 1.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(trainer, is_best=False, normal_save=False):
    """
    Save model checkpoint.
    
    Args:
        trainer: Trainer to save checkpoint for
        is_best: Whether this is the best model so far
        normal_save: Whether this is a normal save
    """
    if hasattr(trainer, 'g_optimizer'):
        checkpoint = {
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
            'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
            'g_scheduler_state_dict': trainer.g_scheduler.state_dict(),
            'd_scheduler_state_dict': trainer.d_scheduler.state_dict(),
            'g_loss': trainer.best_g_loss,
            'd_loss': trainer.best_d_loss,
            'g_ema_shadow': trainer.g_ema.shadow,
            'd_ema_shadow': trainer.d_ema.shadow
        }
    else:
        checkpoint = {
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'loss': trainer.best_loss,
            'ema_shadow': trainer.ema.shadow if trainer.ema else None
        }
    
    if normal_save:
        filepath = os.path.join(trainer.config['checkpoint_dir'], f"checkpoint_epoch_{trainer.current_epoch}.pth")
    elif is_best:
        filepath = os.path.join(trainer.config['checkpoint_dir'], "best_model.pth")

    torch.save(checkpoint, filepath)
    if normal_save:
        print(f"Checkpoint saved to {filepath}")
    elif is_best:
        print(f"Best model saved to {filepath}")


def load_checkpoint(trainer, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        trainer: Trainer to load checkpoint into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    
    trainer.model.load_state_dict(checkpoint['model_state_dict'])

    if hasattr(trainer, 'g_optimizer'):
        trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        trainer.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        trainer.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        trainer.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        trainer.g_ema.shadow = checkpoint['g_ema_shadow']
        trainer.d_ema.shadow = checkpoint['d_ema_shadow']
        trainer.best_g_loss = checkpoint['g_loss']
        trainer.best_d_loss = checkpoint['d_loss']
    else:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.ema.shadow = checkpoint['ema_shadow']
        trainer.best_loss = checkpoint['loss']
  
    trainer.current_epoch = checkpoint['epoch'] + 1
    
    print(f"Resumed from epoch {trainer.current_epoch}")
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")


def create_lr_scheduler(optimizer: torch.optim.Optimizer, 
                       scheduler_type: str = "cosine",
                       num_training_steps: int = 100000,
                       warmup_steps: int = 1000,
                       **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ("cosine", "linear", "step")
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, **kwargs)
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, 
                           total_iters=num_training_steps, **kwargs)
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=kwargs.get('step_size', 1000), 
                          gamma=kwargs.get('gamma', 0.9), **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        Device to use
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_optimizer(model: nn.Module, optimizer_type: str = "adam",
                    learning_rate: float = 1e-4, **kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        learning_rate: Learning rate
        **kwargs: Additional arguments for optimizer
        
    Returns:
        Optimizer
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def log_hyperparameters(config: Dict[str, Any], log_dir: str = "logs"):
    """
    Log hyperparameters to file.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save logs
    """
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def log_metrics(log_dir: str, train_loss, val_loss, epoch):
        """
        Log training metrics.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            epoch: Current epoch
        """
        # Log to file
        log_file = os.path.join(log_dir, "training_log.txt")
        # Create file from scratch if first epoch
        if epoch == 1:
            if os.path.exists(log_file):
                os.remove(log_file)
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n")
        
        # Print to console
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")


def compute_model_size_mb(model: nn.Module) -> float:
    """
    Compute model size in MB.
    
    Args:
        model: Model to compute size for
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: nn.Module):
    """
    Print model summary including parameter count and size.
    
    Args:
        model: Model to summarize
    """
    total_params = count_parameters(model)
    model_size = compute_model_size_mb(model)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  Model structure:")
    print(model)


def create_dataloader(config, device = None, normalize = True, dataset = 'mnist'):
    """
    Create a DataLoader with common settings.
    
    Args:
        trainer: Trainer to create dataloader for
        dataset: Dataset to load    
    Returns:
        DataLoader
    """
    
    # Disable pin_memory on MPS (Apple Silicon) as it's not supported
    if device is not None and device.type == 'mps':
        pin_memory = False
    
    if dataset == 'mnist':
        # Data transformations
        if not normalize:
            transform = transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.ToTensor(),
                # Normalize to [0, 1] for VAE (since we use sigmoid in decoder)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for Tanh output
            ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root=config['data_dir'], train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root=config['data_dir'], train=False, download=True, transform=transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        drop_last=True
    )
    return train_loader, val_loader


def save_image_grid(images, filepath, nrow=4):
        """
        Save a grid of images.
        
        Args:
            images: Tensor of images
            filepath: Path to save the grid
            nrow: Number of images per row
        """
        # Create grid
        grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)
        
        # Save grid
        vutils.save_image(grid, filepath)


def load_config(config_path: str, default_config: dict) -> dict:
    """
    Load configuration from a Python file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Get default config first
    config = default_config
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Update with values from config file if they exist
    if hasattr(config_module, 'DEFAULT_CONFIG'):
        config.update(config_module.DEFAULT_CONFIG)
    
    return config


def plot_losses(log_file: str, save_path: str):
    """
    Plot training and validation losses from training log.
    
    Args:
        log_file: Path to training log file
        save_path: Path to save the plot
    """
    train_losses = []
    val_losses = []
    epochs = []
    
    # Extract losses from log file
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('Epoch'):
                # Parse line like "Epoch X: Train Loss: 0.123456, Val Loss: 0.123456"
                parts = line.strip().split(':')
                epoch = int(parts[0].split()[1])
                train_loss = float(parts[2].split(',')[0])
                val_loss = float(parts[3])
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Test utilities
    import torch.nn as nn
    
    # Test model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Test parameter counting
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test model size
    print(f"Model size: {compute_model_size_mb(model):.2f} MB")
    
    # Test EMA
    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Simulate training step
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    # Update EMA
    ema.update(model)
    
    # Test device detection
    device = get_device()
    print(f"Using device: {device}")
    
    # Test seeding
    set_seed(42)
    print("Seed set to 42") 