import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Dict, Any, Optional
import copy


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


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Any, epoch: int, loss: float, 
                   ema: Optional[EMA] = None, 
                   filepath: str = "checkpoint.pth"):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss
        ema: EMA object (optional)
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'ema_shadow': ema.shadow if ema else None
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Any, ema: Optional[EMA] = None,
                   filepath: str = "checkpoint.pth") -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        ema: EMA object to load state into (optional)
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=get_device())
    
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    
    if ema and checkpoint['ema_shadow']:
        ema.shadow = checkpoint['ema_shadow']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


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


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = True,
                     num_workers: int = 4, pin_memory: bool = True, device=None):
    """
    Create a DataLoader with common settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        device: Device to use (for MPS compatibility)
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    # Disable pin_memory on MPS (Apple Silicon) as it's not supported
    if device is not None and device.type == 'mps':
        pin_memory = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


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