"""
Default configuration for DDPM training on MNIST.
"""

# Model configuration
MODEL_CONFIG = {
    'image_size': 32,
    'in_channels': 1,
    'hidden_dims': [32, 64, 128],
    'latent_dim': 128,
}

# Training configuration
TRAINING_CONFIG = {
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
}

# Diffusion configuration
LOSS_CONFIG = {
    'loss_type': 'vae', # 'vae' or 'beta_vae'
    'reconstruction_loss_type': 'mse', # 'mse' or 'bce'
    'beta': 1.0,
    'beta_start': 0.0,
    'beta_end': 1.0,
    'beta_steps': 1000,
}

# Paths configuration
PATHS_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'sample_dir': 'samples',
    'data_dir': 'data',
    'results_dir': 'evaluation_results',
    'experiment_name': 'test_VAE',
}

# Combine all configurations
DEFAULT_CONFIG = {
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    **LOSS_CONFIG,
    **PATHS_CONFIG,
    'resume_from': None,
    'steps_per_epoch': 469,  # MNIST train size / batch_size
} 