"""
Default configuration for DDPM training on MNIST.
"""

# Model configuration
MODEL_CONFIG = {
    'image_size': 28,
    'in_channels': 1,
    'out_channels': 1,
    'model_channels': 64,
    'time_emb_dim': 128,
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'scheduler_type': 'cosine',
    'warmup_steps': 1000,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'max_grad_norm': 1.0,
    'ema_decay': 0.9999,
    'save_every': 10,
    'sample_every': 5,
    'seed': 42,
}

# Diffusion configuration
DIFFUSION_CONFIG = {
    'num_train_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'linear',
    'loss_type': 'l2',
}

# Paths configuration
PATHS_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'sample_dir': 'samples',
    'data_dir': 'data',
    'results_dir': 'evaluation_results',
}

# Sampling configuration
SAMPLING_CONFIG = {
    'num_inference_steps': 50,
    'eta': 0.0,
    'sampler_type': 'deterministic',
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'num_samples': 1000,
    'batch_size': 32,
    'num_workers': 4,
}

# Combine all configurations
DEFAULT_CONFIG = {
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    **DIFFUSION_CONFIG,
    **PATHS_CONFIG,
    **SAMPLING_CONFIG,
    **EVALUATION_CONFIG,
    'resume_from': None,
    'steps_per_epoch': 469,  # MNIST train size / batch_size
} 