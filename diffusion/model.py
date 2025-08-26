import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusion.utils import count_parameters, get_device


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time steps.
    Maps time steps to a higher dimensional space for better representation.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.device = get_device()

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        
        # Handle both scalar and tensor inputs
        if time.dim() == 0:
            # Scalar input (during sampling)
            time = time.unsqueeze(0)
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    Basic convolutional block with time embedding.
    Consists of two conv layers with group norm and SiLU activation.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()
        
    def forward(self, x, t):
        # First conv
        h = self.relu(self.bnorm1(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend time embeddings to 2D
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time embeddings
        h = h + time_emb
        # Second conv
        h = self.relu(self.bnorm2(self.conv2(h)))
        # Down or up
        return self.transform(h)


class UNetMini(nn.Module):
    """
    UNet-mini architecture for DDPM on MNIST.
    No attention layers, uses ε-prediction.
    """
    def __init__(self, image_size=28, in_channels=1, out_channels=1, model_channels=64, 
                 time_emb_dim=128):
        super().__init__()
        self.device = get_device()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Down sampling
        self.down1 = Block(model_channels, model_channels, time_emb_dim, up=False)
        self.down2 = Block(model_channels, model_channels, time_emb_dim, up=False)
        self.down3 = Block(model_channels, model_channels, time_emb_dim, up=False)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
        )
        
        # Up sampling
        self.up1 = Block(model_channels, model_channels, time_emb_dim, up=True)
        self.up2 = Block(model_channels, model_channels, time_emb_dim, up=True)
        self.up3 = Block(model_channels, model_channels, time_emb_dim, up=True)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 1)
        )

        # Move to device
        self.to(self.device)
        
    def forward(self, x, timesteps):
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Time steps tensor of shape (batch_size,)
            
        Returns:
            Predicted noise tensor of same shape as input
        """
        # Time embedding
        t = self.time_mlp(timesteps)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Down sampling with skip connections
        down1 = self.down1(x, t)
        down2 = self.down2(down1, t)
        down3 = self.down3(down2, t)
        
        # Bottleneck
        bottleneck = self.bottleneck(down3)
        
        # Up sampling with skip connections
        up1 = self.up1(torch.cat([bottleneck, down3], dim=1), t)
        up2 = self.up2(torch.cat([up1, down2], dim=1), t)
        up3 = self.up3(torch.cat([up2, down1], dim=1), t)
        
        # Final output
        return self.final_conv(up3)


if __name__ == "__main__":
    # Test the model
    device = get_device()
    model = UNetMini()
    x = torch.randn(1, 1, 28, 28).to(device)
    timesteps = torch.randint(0, 1000, (1,)).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x, timesteps).shape}") 