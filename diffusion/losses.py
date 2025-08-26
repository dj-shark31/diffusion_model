import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from diffusion.utils import get_device

class DDPMLoss(nn.Module):
    """
    DDPM Loss function for epsilon prediction.
    Implements the simplified objective from the DDPM paper.
    """
    
    def __init__(self, loss_type: str = "l2"):
        """
        Initialize the DDPM loss.
        
        Args:
            loss_type: Type of loss ("l2", "l1", "huber")
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, model_output: torch.FloatTensor, 
                target: torch.FloatTensor, 
                timesteps: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Compute the DDPM loss.
        
        Args:
            model_output: Predicted noise from the model
            target: Target noise (ground truth)
            timesteps: Timesteps (not used in basic DDPM loss)
            
        Returns:
            Loss value
        """
        return self.loss_fn(model_output, target)


class WeightedDDPMLoss(nn.Module):
    """
    Weighted DDPM Loss with timestep-dependent weighting.
    Can be used to focus on specific timesteps during training.
    """
    
    def __init__(self, loss_type: str = "l2", weight_schedule: str = "uniform"):
        """
        Initialize the weighted DDPM loss.
        
        Args:
            loss_type: Type of loss ("l2", "l1", "huber")
            weight_schedule: Weighting schedule ("uniform", "linear", "cosine")
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight_schedule = weight_schedule

        if loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_timestep_weights(self, timesteps: torch.LongTensor, 
                           num_train_timesteps: int = 1000) -> torch.FloatTensor:
        """
        Get weights for each timestep based on the schedule.
        
        Args:
            timesteps: Current timesteps
            num_train_timesteps: Total number of training timesteps
            
        Returns:
            Weights for each timestep
        """
        if self.weight_schedule == "uniform":
            return torch.ones_like(timesteps, dtype=torch.float)
        elif self.weight_schedule == "linear":
            # Linear weighting: higher weights for later timesteps
            return timesteps.float() / num_train_timesteps
        elif self.weight_schedule == "cosine":
            # Cosine weighting: higher weights for middle timesteps
            return torch.cos(timesteps.float() * torch.pi / num_train_timesteps)
        else:
            raise ValueError(f"Unknown weight schedule: {self.weight_schedule}")
    
    def forward(self, model_output: torch.FloatTensor, 
                target: torch.FloatTensor, 
                timesteps: torch.LongTensor,
                num_train_timesteps: int = 1000) -> torch.FloatTensor:
        """
        Compute the weighted DDPM loss.
        
        Args:
            model_output: Predicted noise from the model
            target: Target noise (ground truth)
            timesteps: Timesteps for weighting
            num_train_timesteps: Total number of training timesteps
            
        Returns:
            Weighted loss value
        """
        # Compute per-pixel loss
        loss = self.loss_fn(model_output, target)
        
        # Average over spatial dimensions
        loss = loss.mean(dim=[1, 2, 3])
        
        # Get timestep weights
        weights = self.get_timestep_weights(timesteps, num_train_timesteps)
        
        # Apply weights and average over batch
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss


class VLBoundLoss(nn.Module):
    """
    Variational Lower Bound Loss for DDPM.
    Implements the full VLB objective from the DDPM paper.
    """
    
    def __init__(self, scheduler):
        """
        Initialize the VLB loss.
        
        Args:
            scheduler: DDPM scheduler for computing VLB terms
        """
        super().__init__()
        self.scheduler = scheduler
    
    def forward(self, model_output: torch.FloatTensor, 
                target: torch.FloatTensor, 
                timesteps: torch.LongTensor,
                original_samples: torch.FloatTensor,
                noisy_samples: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the VLB loss.
        
        Args:
            model_output: Predicted noise from the model
            target: Target noise (ground truth)
            timesteps: Current timesteps
            original_samples: Original clean samples
            noisy_samples: Noisy samples at current timestep
            
        Returns:
            VLB loss value
        """
        # Get scheduler values for current timesteps
        alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        alpha_t_prev = self.scheduler.alphas_cumprod_prev[timesteps].view(-1, 1, 1, 1)
        beta_t = self.scheduler.betas[timesteps].view(-1, 1, 1, 1)
        
        # Compute predicted x_0
        pred_original_sample = (noisy_samples - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # Compute VLB terms
        # Term 1: KL divergence between q(x_{t-1}|x_t, x_0) and p(x_{t-1}|x_t)
        kl_div = self._compute_kl_div(original_samples, pred_original_sample, 
                                    alpha_t, alpha_t_prev, beta_t)
        
        # Term 2: Reconstruction loss for x_0
        recon_loss = F.mse_loss(pred_original_sample, original_samples, reduction='none')
        recon_loss = recon_loss.mean(dim=[1, 2, 3])
        
        # Combine terms
        vlb_loss = kl_div + recon_loss
        
        return vlb_loss.mean()
    
    def _compute_kl_div(self, original_samples: torch.FloatTensor,
                       pred_original_sample: torch.FloatTensor,
                       alpha_t: torch.FloatTensor,
                       alpha_t_prev: torch.FloatTensor,
                       beta_t: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute KL divergence between q(x_{t-1}|x_t, x_0) and p(x_{t-1}|x_t).
        """
        # Compute means
        q_mean = (torch.sqrt(alpha_t_prev) * beta_t * original_samples + 
                 torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) * pred_original_sample) / (1 - alpha_t)
        
        p_mean = (torch.sqrt(alpha_t_prev) * beta_t * pred_original_sample + 
                 torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) * pred_original_sample) / (1 - alpha_t)
        
        # Compute variances
        q_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
        p_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
        
        # Compute KL divergence
        kl_div = 0.5 * (torch.log(p_var / q_var) + (q_var + (q_mean - p_mean) ** 2) / p_var - 1)
        kl_div = kl_div.mean(dim=[1, 2, 3])
        
        return kl_div


def compute_loss(model_output: torch.FloatTensor, 
                target: torch.FloatTensor, 
                loss_type: str = "l2",
                timesteps: Optional[torch.LongTensor] = None,
                scheduler: Optional[object] = None,
                original_samples: Optional[torch.FloatTensor] = None,
                noisy_samples: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Convenience function to compute DDPM loss.
    
    Args:
        model_output: Predicted noise from the model
        target: Target noise (ground truth)
        loss_type: Type of loss ("l2", "l1", "huber", "weighted", "vlb")
        timesteps: Timesteps (required for weighted and vlb losses)
        scheduler: DDPM scheduler (required for vlb loss)
        original_samples: Original clean samples (required for vlb loss)
        noisy_samples: Noisy samples (required for vlb loss)
        
    Returns:
        Loss value
    """
    if loss_type in ["l2", "l1", "huber"]:
        loss_fn = DDPMLoss(loss_type)
        return loss_fn(model_output, target, timesteps)
    elif loss_type == "weighted":
        loss_fn = WeightedDDPMLoss()
        return loss_fn(model_output, target, timesteps)
    elif loss_type == "vlb":
        if scheduler is None or original_samples is None or noisy_samples is None:
            raise ValueError("scheduler, original_samples, and noisy_samples required for VLB loss")
        loss_fn = VLBoundLoss(scheduler)
        return loss_fn(model_output, target, timesteps, original_samples, noisy_samples)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss functions
    batch_size = 4
    model_output = torch.randn(batch_size, 1, 28, 28)
    target = torch.randn(batch_size, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Test basic loss
    loss_fn = DDPMLoss("l2")
    loss = loss_fn(model_output, target)
    print(f"Basic L2 loss: {loss.item():.4f}")
    
    # Test weighted loss
    weighted_loss_fn = WeightedDDPMLoss("l2", "linear")
    weighted_loss = weighted_loss_fn(model_output, target, timesteps)
    print(f"Weighted loss: {weighted_loss.item():.4f}") 