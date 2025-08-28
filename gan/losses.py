"""
Loss functions for Generative Adversarial Networks (GANs).

This module provides the loss functions used in GAN training,
including binary cross entropy and Wasserstein loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GANLoss(nn.Module):
    """
    Standard GAN Loss using Binary Cross Entropy.
    
    The discriminator tries to maximize log(D(x)) + log(1-D(G(z)))
    The generator tries to minimize log(1-D(G(z)))
    """
    
    def __init__(self, label_smoothing: float = 0.1):
        """
        Initialize GAN loss.
        
        Args:
            label_smoothing: Amount of label smoothing to apply
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce_loss = nn.BCELoss()
        
    def discriminator_loss(self, real_probs: torch.Tensor, fake_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_probs: Probabilities for real images
            fake_probs: Probabilities for fake images
            
        Returns:
            Discriminator loss
        """
        batch_size = real_probs.size(0)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=real_probs.device) * (1 - self.label_smoothing)
        fake_labels = torch.zeros(batch_size, 1, device=fake_probs.device)
        
        # Loss for real images (should be classified as real)
        real_loss = self.bce_loss(real_probs, real_labels)
        
        # Loss for fake images (should be classified as fake)
        fake_loss = self.bce_loss(fake_probs, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        
        return d_loss
    
    def generator_loss(self, fake_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_probs: Probabilities for fake images
            
        Returns:
            Generator loss
        """
        batch_size = fake_probs.size(0)
        
        # Labels for fake images (generator wants them to be classified as real)
        fake_labels = torch.ones(batch_size, 1, device=fake_probs.device)
        
        # Loss for fake images (should be classified as real)
        g_loss = self.bce_loss(fake_probs, fake_labels)
        
        return g_loss
    
    def forward(self, real_probs: torch.Tensor, fake_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both discriminator and generator losses.
        
        Args:
            real_probs: Probabilities for real images
            fake_probs: Probabilities for fake images
            
        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        if real_probs is not None and fake_probs is not None:
            d_loss = self.discriminator_loss(real_probs, fake_probs)
        else:
            d_loss = None
        if fake_probs is not None:
            g_loss = self.generator_loss(fake_probs)
        else:
            g_loss = None
        
        return d_loss, g_loss


class WassersteinLoss(nn.Module):
    """
    Wasserstein GAN Loss.
    
    The discriminator tries to maximize D(x) - D(G(z))
    The generator tries to minimize -D(G(z))
    """
    
    def __init__(self):
        """
        Initialize Wasserstein loss.
        """
        super().__init__()
        
    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_scores: Scores for real images (before sigmoid)
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Discriminator loss
        """
        # Discriminator wants to maximize D(x) - D(G(z))
        d_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))
        
        return d_loss
    
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Generator loss
        """
        # Generator wants to minimize -D(G(z))
        g_loss = -torch.mean(fake_scores)
        
        return g_loss
    
    def forward(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both discriminator and generator losses.
        
        Args:
            real_scores: Scores for real images (before sigmoid)
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        if real_scores is not None and fake_scores is not None:
            d_loss = self.discriminator_loss(real_scores, fake_scores)
        else:
            d_loss = None
        if fake_scores is not None:
            g_loss = self.generator_loss(fake_scores)
        else:
            g_loss = None
        
        return d_loss, g_loss


class HingeLoss(nn.Module):
    """
    Hinge Loss for GANs.
    
    The discriminator tries to maximize min(0, -1 + D(x)) + min(0, -1 - D(G(z)))
    The generator tries to minimize -D(G(z))
    """
    
    def __init__(self):
        """
        Initialize hinge loss.
        """
        super().__init__()
        
    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_scores: Scores for real images (before sigmoid)
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Discriminator loss
        """
        # Hinge loss for discriminator
        real_loss = torch.mean(F.relu(1.0 - real_scores))
        fake_loss = torch.mean(F.relu(1.0 + fake_scores))
        
        d_loss = real_loss + fake_loss
        
        return d_loss
    
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Generator loss
        """
        # Hinge loss for generator
        g_loss = -torch.mean(fake_scores)
        
        return g_loss
    
    def forward(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both discriminator and generator losses.
        
        Args:
            real_scores: Scores for real images (before sigmoid)
            fake_scores: Scores for fake images (before sigmoid)
            
        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        if real_scores is not None and fake_scores is not None:
            d_loss = self.discriminator_loss(real_scores, fake_scores)
        else:
            d_loss = None
        if fake_scores is not None:
            g_loss = self.generator_loss(fake_scores)
        else:
            g_loss = None
        
        return d_loss, g_loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss for GANs.
    
    Encourages the generator to produce images that have similar
    features to real images when passed through the discriminator.
    """
    
    def __init__(self, feature_weight: float = 1.0):
        """
        Initialize feature matching loss.
        
        Args:
            feature_weight: Weight for feature matching loss
        """
        super().__init__()
        self.feature_weight = feature_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: Features from real images
            fake_features: Features from fake images
            
        Returns:
            Feature matching loss
        """
        return self.feature_weight * self.mse_loss(fake_features, real_features)

if __name__ == "__main__":
    # Test GAN loss functions
    batch_size = 4
    
    # Create dummy data
    real_probs = torch.rand(batch_size, 1)
    fake_probs = torch.rand(batch_size, 1)
    real_scores = torch.randn(batch_size, 1)
    fake_scores = torch.randn(batch_size, 1)
    real_features = torch.randn(batch_size, 512)
    fake_features = torch.randn(batch_size, 512)
    
    # Test standard GAN loss
    gan_loss = GANLoss(label_smoothing=0.1)
    d_loss, g_loss = gan_loss(real_probs, fake_probs)
    
    print(f"Standard GAN Loss:")
    print(f"  Discriminator loss: {d_loss:.4f}")
    print(f"  Generator loss: {g_loss:.4f}")
    
    # Test Wasserstein loss
    wgan_loss = WassersteinLoss()
    d_loss, g_loss = wgan_loss(real_scores, fake_scores)
    
    print(f"\nWasserstein Loss:")
    print(f"  Discriminator loss: {d_loss:.4f}")
    print(f"  Generator loss: {g_loss:.4f}")
    
    # Test hinge loss
    hinge_loss = HingeLoss()
    d_loss, g_loss = hinge_loss(real_scores, fake_scores)
    
    print(f"\nHinge Loss:")
    print(f"  Discriminator loss: {d_loss:.4f}")
    print(f"  Generator loss: {g_loss:.4f}")
    
    # Test feature matching loss
    feature_loss = FeatureMatchingLoss(feature_weight=1.0)
    fm_loss = feature_loss(real_features, fake_features)
    
    print(f"\nFeature Matching Loss:")
    print(f"  Feature matching loss: {fm_loss:.4f}")
