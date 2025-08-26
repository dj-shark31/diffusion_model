import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from diffusion.utils import get_device

class DDPMScheduler:
    """
    DDPM Scheduler with linear beta schedule.
    Implements the forward and reverse diffusion processes.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, 
                 beta_end: float = 0.02, beta_schedule: str = "linear"):
        """
        Initialize the DDPM scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion timesteps (T=1000)
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            beta_schedule: Type of beta schedule ("linear", "cosine", etc.)
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = get_device()
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(self.device)
        elif beta_schedule == "uniform":
            self.betas = torch.ones(num_train_timesteps) * beta_end.to(self.device)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps).to(self.device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for reverse process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps):
        """Cosine beta schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples: torch.FloatTensor, 
                  timesteps: torch.LongTensor) -> torch.FloatTensor:
        """
        Add noise to the original samples according to the forward process.
        
        Args:
            original_samples: Original clean images
            timesteps: Timesteps for each sample
            
        Returns:
            Noisy images
        """
        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples, device=self.device)
        noisy_samples = sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.FloatTensor, timestep: int, 
             sample: torch.FloatTensor, eta: float = 0.0) -> torch.FloatTensor:
        """
        Perform a single reverse diffusion step.
        
        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            eta: Controls the amount of noise (0 = deterministic, 1 = stochastic)
            
        Returns:
            Denoised sample
        """
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=self.device)
        
        # Get the values for the current timestep
        alpha_t = self.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        alpha_t_prev = self.alphas_cumprod_prev[timestep].view(-1, 1, 1, 1)
        beta_t = self.betas[timestep].view(-1, 1, 1, 1)
        
        # Predict x_0 from the model output (epsilon prediction)
        pred_original_sample = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # Compute the mean of the posterior
        pred_sample_direction = torch.sqrt(1 - alpha_t_prev) * model_output
        mean = torch.sqrt(alpha_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[timestep].view(-1, 1, 1, 1)
            std = torch.sqrt(variance) * eta
            mean = mean + std * noise
        
        return mean
    
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input.
        For DDPM, this is a no-op.
        """
        return sample
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain.
        For DDPM, we use all timesteps.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, 
                                    self.num_train_timesteps // num_inference_steps).to(self.device)

    def __len__(self):
        return self.num_train_timesteps


class DDIMScheduler(DDPMScheduler):
    """
    DDIM Scheduler for deterministic sampling.
    Extends DDPM scheduler with DDIM sampling capabilities.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, 
                 beta_end: float = 0.02, beta_schedule: str = "linear", eta: float = 0.0):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.eta = eta
    
    def step(self, model_output: torch.FloatTensor, timestep: int, 
             sample: torch.FloatTensor, eta: Optional[float] = None) -> torch.FloatTensor:
        """
        DDIM step for deterministic sampling.
        """
        if eta is None:
            eta = self.eta
        
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=self.device)
        
        # Get the values for the current timestep
        alpha_t = self.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        alpha_t_prev = self.alphas_cumprod_prev[timestep].view(-1, 1, 1, 1)
        
        # Predict x_0 from the model output
        pred_original_sample = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # DDIM formula
        pred_epsilon = model_output
        
        # Compute the mean
        pred_sample_direction = torch.sqrt(1 - alpha_t_prev) * pred_epsilon
        mean = torch.sqrt(alpha_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[timestep].view(-1, 1, 1, 1)
            std = torch.sqrt(variance) * eta
            mean = mean + std * noise
        
        return mean


if __name__ == "__main__":
    # Test the scheduler
    scheduler = DDPMScheduler()
    print(f"Number of timesteps: {scheduler.num_train_timesteps}")
    print(f"Beta schedule shape: {scheduler.betas.shape}")
    print(f"Alphas cumprod shape: {scheduler.alphas_cumprod.shape}")
    
    # Test adding noise
    x = torch.randn(2, 1, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))
    noisy_x, noise = scheduler.add_noise(x, timesteps)
    print(f"Original shape: {x.shape}")
    print(f"Noisy shape: {noisy_x.shape}")
    print(f"Noise shape: {noise.shape}") 