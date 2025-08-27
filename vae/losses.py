"""
Loss functions for Variational Autoencoder (VAE).

This module provides the loss functions used in VAE training,
including reconstruction loss and KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAELoss(nn.Module):
    """
    VAE Loss function combining reconstruction loss and KL divergence.
    
    The total loss is: L = L_recon + β * L_KL
    where β controls the weight of the KL divergence term.
    """
    
    def __init__(self, beta: float = 1.0, loss_type: str = "mse"):
        """
        Initialize VAE loss.
        
        Args:
            beta: Weight for KL divergence term (β-VAE)
            loss_type: Type of reconstruction loss ("mse" or "bce")
        """
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        
    def reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            
        Returns:
            Reconstruction loss
        """
        if self.loss_type == "mse":
            # Mean Squared Error loss
            return F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        elif self.loss_type == "bce":
            # Binary Cross Entropy loss (for binary images)
            return F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between learned distribution and prior.
        
        Args:
            mu: Mean of learned distribution
            log_var: Log variance of learned distribution
            
        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss / mu.size(0)
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, 
                mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total VAE loss.
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(recon_x, x)
        
        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, log_var)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class BetaVAELoss(VAELoss):
    """
    β-VAE Loss with annealing schedule for β.
    
    Gradually increases β during training to encourage
    better disentanglement of latent factors.
    """
    
    def __init__(self, beta_start: float = 0.0, beta_end: float = 1.0, 
                 beta_steps: int = 1000, loss_type: str = "mse"):
        """
        Initialize β-VAE loss with annealing.
        
        Args:
            beta_start: Starting value for β
            beta_end: Final value for β
            beta_steps: Number of steps to anneal β
            loss_type: Type of reconstruction loss
        """
        super().__init__(beta=beta_start, loss_type=loss_type)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.current_step = 0
        
    def update_beta(self, step: int):
        """
        Update β value based on current training step.
        
        Args:
            step: Current training step
        """
        self.current_step = step
        if step < self.beta_steps:
            # Linear annealing
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * step / self.beta_steps
        else:
            self.beta = self.beta_end


class DisentangledVAELoss(VAELoss):
    """
    Disentangled VAE Loss with additional regularization terms.
    
    Adds total correlation penalty to encourage disentanglement
    of latent factors.
    """
    
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, loss_type: str = "mse"):
        """
        Initialize disentangled VAE loss.
        
        Args:
            beta: Weight for KL divergence
            gamma: Weight for total correlation penalty
            loss_type: Type of reconstruction loss
        """
        super().__init__(beta=beta, loss_type=loss_type)
        self.gamma = gamma
        
    def total_correlation(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute total correlation penalty.
        
        Args:
            z: Sampled latent vectors
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Total correlation penalty
        """
        # Compute log q(z) (joint distribution)
        log_qz = self._log_density_gaussian(z, mu, log_var)
        
        # Compute log prod q(z_i) (product of marginals)
        # For each dimension i, compute the marginal log density
        log_qz_marginals = []
        for i in range(z.shape[1]):  # For each latent dimension
            # Extract mean and variance for dimension i
            mu_i = mu[:, i:i+1]
            log_var_i = log_var[:, i:i+1]
            z_i = z[:, i:i+1]
            
            # Compute log density for dimension i
            log_density_i = -0.5 * (log_var_i + (z_i - mu_i).pow(2) / log_var_i.exp())
            log_qz_marginals.append(log_density_i)
        
        # Sum over all marginal log densities
        log_qz_prod = torch.cat(log_qz_marginals, dim=1).sum(dim=1)
        
        # Total correlation = KL(q(z) || prod q(z_i))
        tc = torch.sum(log_qz - log_qz_prod) / z.size(0)
        
        return tc
    
    def _log_density_gaussian(self, z: torch.Tensor, mu: torch.Tensor, 
                             log_var: torch.Tensor, sum_dims: bool = True) -> torch.Tensor:
        """
        Compute log density of Gaussian distribution.
        
        Args:
            z: Points to evaluate
            mu: Mean of distribution
            log_var: Log variance of distribution
            sum_dims: Whether to sum over dimensions
            
        Returns:
            Log density
        """
        log_density = -0.5 * (log_var + (z - mu).pow(2) / log_var.exp())
        
        if sum_dims:
            return torch.sum(log_density, dim=1)
        else:
            return log_density
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, 
                mu: torch.Tensor, log_var: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total disentangled VAE loss.
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            z: Sampled latent vectors
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss, tc_loss)
        """
        # Compute base VAE loss
        total_loss, recon_loss, kl_loss = super().forward(recon_x, x, mu, log_var)
        
        # Compute total correlation penalty
        tc_loss = self.total_correlation(z, mu, log_var)
        
        # Add total correlation penalty
        total_loss = total_loss + self.gamma * tc_loss
        
        return total_loss, recon_loss, kl_loss, tc_loss


if __name__ == "__main__":
    # Test VAE loss functions
    batch_size = 4
    image_size = 28
    latent_dim = 128
    
    # Create dummy data
    x = torch.randn(batch_size, 1, image_size, image_size)
    recon_x = torch.randn(batch_size, 1, image_size, image_size)
    mu = torch.randn(batch_size, latent_dim)
    log_var = torch.randn(batch_size, latent_dim)
    z = torch.randn(batch_size, latent_dim)
    
    # Test standard VAE loss
    vae_loss = VAELoss(beta=1.0, loss_type="mse")
    total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)
    
    print(f"Standard VAE Loss:")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Reconstruction loss: {recon_loss:.4f}")
    print(f"  KL loss: {kl_loss:.4f}")
    
    # Test β-VAE loss
    beta_vae_loss = BetaVAELoss(beta_start=0.0, beta_end=1.0, beta_steps=1000)
    beta_vae_loss.update_beta(500)
    total_loss, recon_loss, kl_loss = beta_vae_loss(recon_x, x, mu, log_var)
    
    print(f"\nβ-VAE Loss (step 500):")
    print(f"  β: {beta_vae_loss.beta:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Reconstruction loss: {recon_loss:.4f}")
    print(f"  KL loss: {kl_loss:.4f}")
    
    # Test disentangled VAE loss
    disentangled_loss = DisentangledVAELoss(beta=1.0, gamma=1.0)
    total_loss, recon_loss, kl_loss, tc_loss = disentangled_loss(recon_x, x, mu, log_var, z)
    
    print(f"\nDisentangled VAE Loss:")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Reconstruction loss: {recon_loss:.4f}")
    print(f"  KL loss: {kl_loss:.4f}")
    print(f"  Total correlation loss: {tc_loss:.4f}")
