"""
Default configuration for DDPM training on MNIST.
"""

# Model configuration
MODEL_CONFIG = {
    'image_size': 32,
    'in_channels': 1,
    'out_channels': 1,
    'model_channels': 64,
    'time_emb_dim': 128,
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'scheduler_type': 'cosine',
    'warmup_steps': 200,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'optimizer_type': 'adamw',
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
    'experiment_name': 'test_2',
}

# Sampling configuration
SAMPLING_CONFIG = {
    'num_inference_steps': 1000,
    'eta': 1.0,
    'sampler_type': 'ancestral',
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'num_samples': 1000,
    'batch_size': 32,
    'num_workers': 8,
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
    'steps_per_epoch': 938,  # MNIST train size (60,000) / batch_size
} 