import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
import tqdm
from diffusion.utils import get_device


class AncestralSampler:
    """
    Ancestral sampling for DDPM.
    Implements the standard DDPM sampling procedure.
    Highest quality sampler, but slowest.
    """
    
    def __init__(self, scheduler, num_inference_steps: int = 1000, eta: float = 1.0):
        """
        Initialize the ancestral sampler.
        
        Args:
            scheduler: DDPM scheduler
            num_inference_steps: Number of inference steps
            eta: Controls the amount of noise (0 = deterministic, 1 = stochastic)
        """
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)
        self.device = self.scheduler.device
        self.eta = eta

    def step(self, model_output: torch.FloatTensor, timestep: int, 
             sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        Perform a single ancestral sampling step.
        
        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            
        Returns:
            Denoised sample
        """
        return self.scheduler.step(model_output, timestep, sample, eta=self.eta)
    
    def __call__(self, model, shape: Tuple[int, ...], 
                 progress_bar: bool = True) -> torch.FloatTensor:
        """
        Generate samples using ancestral sampling.
        
        Args:
            model: DDPM model
            shape: Shape of samples to generate (batch_size, channels, height, width)
            device: Device to generate samples on
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        sample = torch.randn(shape, device=self.device)
        
        # Reverse diffusion process
        timesteps = self.scheduler.timesteps
        
        if progress_bar:
            timesteps = tqdm.tqdm(timesteps, desc="Ancestral Sampling")
        
        for timestep in timesteps:
            # Predict noise
            with torch.no_grad():
                model_output = model(sample, timestep)
            
            # Denoise step
            sample = self.step(model_output, timestep, sample)
        
        return sample


class DDIMSampler:
    """
    DDIM Sampler for fast deterministic sampling.
    Implements the DDIM sampling algorithm.
    Much faster than ancestral sampling, and still good quality.
    """
    
    def __init__(self, scheduler, num_inference_steps: int = 50, eta: float = 0.0):
        """
        Initialize the DDIM sampler.
        
        Args:
            scheduler: DDPM scheduler
            num_inference_steps: Number of inference steps (can be much smaller than training steps)
            eta: Controls the amount of noise (0 = deterministic, 1 = stochastic)
        """
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.device = self.scheduler.device
        # Set timesteps for DDIM
        self.scheduler.set_timesteps(num_inference_steps)

    def step(self, model_output: torch.FloatTensor, timestep: int, 
             sample: torch.FloatTensor, prev_timestep: int) -> torch.FloatTensor:
        """
        Perform a single DDIM step.
        
        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            prev_timestep: Previous timestep
            
        Returns:
            Denoised sample
        """
        # Get scheduler values
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        alpha_t_prev = self.scheduler.alphas_cumprod_prev[timestep]
        
        # Predict x_0
        pred_original_sample = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # DDIM formula
        pred_sample_direction = torch.sqrt(1 - alpha_t_prev) * model_output
        mean = torch.sqrt(alpha_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn_like(sample)
            variance = self.scheduler.posterior_variance[timestep]
            std = torch.sqrt(variance) * self.eta
            mean = mean + std * noise
        
        return mean
    
    def __call__(self, model, shape: Tuple[int, ...], 
                 progress_bar: bool = True) -> torch.FloatTensor:
        """
        Generate samples using DDIM sampling.
        
        Args:
            model: DDPM model
            shape: Shape of samples to generate (batch_size, channels, height, width)
            device: Device to generate samples on
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        sample = torch.randn(shape, device=self.device)
        
        # Reverse diffusion process
        original_timesteps = self.scheduler.timesteps
        
        if progress_bar:
            timesteps = tqdm.tqdm(original_timesteps, desc="DDIM Sampling")
        else:
            timesteps = original_timesteps
        
        for i, timestep in enumerate(timesteps):
            # Get previous timestep
            prev_timestep = original_timesteps[i + 1] if i + 1 < len(original_timesteps) else 0
            
            # Predict noise
            with torch.no_grad():
                model_output = model(sample, timestep)
            
            # DDIM step
            sample = self.step(model_output, timestep, sample, prev_timestep)
        
        return sample


def create_sampler(sampler_type: str, scheduler, **kwargs):
    """
    Factory function to create samplers.
    
    Args:
        sampler_type: Type of sampler ("ancestral", "ddim")
        scheduler: DDPM scheduler
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Sampler instance
    """
    if sampler_type == "ancestral":
        return AncestralSampler(scheduler, **kwargs)
    elif sampler_type == "ddim":
        return DDIMSampler(scheduler, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


if __name__ == "__main__":
    # Test the samplers
    from diffusion.schedulers import DDPMScheduler
    from diffusion.model import UNetMini
    from diffusion.utils import get_device
    
    device = get_device()
    
    # Create scheduler and model
    scheduler = DDPMScheduler()
    model = UNetMini()
    
    # Test ancestral sampler
    ancestral_sampler = AncestralSampler(scheduler, num_inference_steps=10)
    samples = ancestral_sampler(model, (2, 1, 28, 28), progress_bar=False)
    print(f"Ancestral samples shape: {samples.shape}")
    
    # Test DDIM sampler
    ddim_sampler = DDIMSampler(scheduler, num_inference_steps=10)
    samples = ddim_sampler(model, (2, 1, 28, 28), progress_bar=False)
    print(f"DDIM samples shape: {samples.shape}") 