import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
import math
from diffusion.utils import get_device

class ClassConditioner(nn.Module):
    """
    Class conditioning for DDPM.
    Embeds class labels and injects them into the model.
    """
    
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        """
        Initialize the class conditioner.
        
        Args:
            num_classes: Number of classes
            embedding_dim: Dimension of class embeddings
        """
        super().__init__()
        self.device = get_device()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.to(self.device)
    
    def forward(self, class_labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Generate class embeddings.
        
        Args:
            class_labels: Class labels of shape (batch_size,)
            
        Returns:
            Class embeddings of shape (batch_size, embedding_dim)
        """
        # Embed class labels
        embeddings = self.class_embedding(class_labels)
        
        # Project embeddings
        embeddings = self.projection(embeddings)
        
        return embeddings


class MaskConditioner(nn.Module):
    """
    Mask conditioning for inpainting tasks.
    Handles masked regions in images.
    """
    
    def __init__(self, mask_channels: int = 1):
        """
        Initialize the mask conditioner.
        
        Args:
            mask_channels: Number of mask channels
        """
        super().__init__()
        self.mask_channels = mask_channels
        self.device = get_device()

    def forward(self, images: torch.FloatTensor, 
                masks: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply mask conditioning to images.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
            masks: Binary masks of shape (batch_size, mask_channels, height, width)
            
        Returns:
            Masked images
        """
        # Ensure masks are binary
        masks = (masks > 0.5).float()
        
        # Apply masks to images
        masked_images = images * (1 - masks)
        
        return masked_images
    
    def create_random_mask(self, image_shape: Tuple[int, ...], 
                          mask_ratio: float = 0.5) -> torch.FloatTensor:
        """
        Create random rectangular masks.
        
        Args:
            image_shape: Shape of images (batch_size, channels, height, width)
            mask_ratio: Ratio of image to mask
            
        Returns:
            Random masks
        """
        batch_size, channels, height, width = image_shape
        
        # Create masks
        masks = torch.zeros(batch_size, self.mask_channels, height, width, device=self.device)
        
        for i in range(batch_size):
            # Random mask dimensions
            mask_h = int(height * mask_ratio)
            mask_w = int(width * mask_ratio)
            
            # Random mask position
            h_start = torch.randint(0, height - mask_h + 1, (1,), device=self.device).item()
            w_start = torch.randint(0, width - mask_w + 1, (1,), device=self.device).item()
            
            # Apply mask
            masks[i, :, h_start:h_start + mask_h, w_start:w_start + mask_w] = 1.0
        
        return masks


class TextConditioner(nn.Module):
    """
    Simple text conditioning using character-level embeddings.
    For more sophisticated text conditioning, consider using transformers.
    """
    
    def __init__(self, vocab_size: int = 128, embedding_dim: int = 128, 
                 max_length: int = 50):
        """
        Initialize the text conditioner.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of text embeddings
            max_length: Maximum text length
        """
        super().__init__()
        self.device = get_device()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Character embeddings
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.to(self.device)
    
    def encode_text(self, text: List[str]) -> torch.FloatTensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings
            
        Returns:
            Text embeddings of shape (batch_size, embedding_dim)
        """
        batch_size = len(text)
        
        # Convert text to character indices
        encoded = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=self.device)
        
        for i, t in enumerate(text):
            # Simple character encoding (ASCII)
            chars = [ord(c) % self.vocab_size for c in t[:self.max_length]]
            encoded[i, :len(chars)] = torch.tensor(chars, device=self.device)
        
        # Create position indices
        positions = torch.arange(self.max_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        char_emb = self.char_embedding(encoded)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        embeddings = char_emb + pos_emb
        
        # Apply transformer
        embeddings = self.transformer(embeddings)
        
        # Global average pooling
        embeddings = embeddings.mean(dim=1)
        
        # Project to final embedding
        embeddings = self.output_projection(embeddings)
        
        return embeddings
    
    def forward(self, text: List[str]) -> torch.FloatTensor:
        """
        Forward pass for text conditioning.
        
        Args:
            text: List of text strings
            
        Returns:
            Text embeddings
        """
        return self.encode_text(text)


class MultiConditioner(nn.Module):
    """
    Combines multiple conditioning types.
    """
    
    def __init__(self, class_conditioner: Optional[ClassConditioner] = None,
                 mask_conditioner: Optional[MaskConditioner] = None,
                 text_conditioner: Optional[TextConditioner] = None,
                 combined_dim: int = 128):
        """
        Initialize the multi-conditioner.
        
        Args:
            class_conditioner: Class conditioner
            mask_conditioner: Mask conditioner
            text_conditioner: Text conditioner
            combined_dim: Dimension of combined embeddings
        """
        super().__init__()
        self.device = get_device()
        self.class_conditioner = class_conditioner
        self.mask_conditioner = mask_conditioner
        self.text_conditioner = text_conditioner
        
        # Calculate total embedding dimension
        total_dim = 0
        if class_conditioner is not None:
            total_dim += class_conditioner.embedding_dim
        if text_conditioner is not None:
            total_dim += text_conditioner.embedding_dim
        
        # Projection to combined dimension
        if total_dim > 0:
            self.combine_projection = nn.Linear(total_dim, combined_dim)
        else:
            self.combine_projection = None
        
        self.combined_dim = combined_dim

        self.to(self.device)
    
    def forward(self, images: Optional[torch.FloatTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                masks: Optional[torch.FloatTensor] = None,
                text: Optional[List[str]] = None) -> Tuple[Optional[torch.FloatTensor], 
                                                          Optional[torch.FloatTensor]]:
        """
        Apply multiple conditioning types.
        
        Args:
            images: Input images
            class_labels: Class labels
            masks: Binary masks
            text: Text descriptions
            
        Returns:
            Tuple of (conditioned_images, combined_embeddings)
        """
        conditioned_images = images
        embeddings = []
        
        # Apply mask conditioning
        if self.mask_conditioner is not None and masks is not None and images is not None:
            conditioned_images = self.mask_conditioner(images, masks)
        
        # Get class embeddings
        if self.class_conditioner is not None and class_labels is not None:
            class_emb = self.class_conditioner(class_labels)
            embeddings.append(class_emb)
        
        # Get text embeddings
        if self.text_conditioner is not None and text is not None:
            text_emb = self.text_conditioner(text)
            embeddings.append(text_emb)
        
        # Combine embeddings
        combined_embeddings = None
        if embeddings:
            combined_embeddings = torch.cat(embeddings, dim=1)
            if self.combine_projection is not None:
                combined_embeddings = self.combine_projection(combined_embeddings)
        
        return conditioned_images, combined_embeddings


def create_conditioner(conditioner_type: str, **kwargs):
    """
    Factory function to create conditioners.
    
    Args:
        conditioner_type: Type of conditioner ("class", "mask", "text", "multi")
        **kwargs: Arguments for the conditioner
        
    Returns:
        Conditioner instance
    """
    if conditioner_type == "class":
        return ClassConditioner(**kwargs)
    elif conditioner_type == "mask":
        return MaskConditioner(**kwargs)
    elif conditioner_type == "text":
        return TextConditioner(**kwargs)
    elif conditioner_type == "multi":
        return MultiConditioner(**kwargs)
    else:
        raise ValueError(f"Unknown conditioner type: {conditioner_type}")


if __name__ == "__main__":
    # Test the conditioners
    batch_size = 4
    device = get_device()
    # Test class conditioner
    class_conditioner = ClassConditioner(num_classes=10)
    class_labels = torch.randint(0, 10, (batch_size,), device=device)
    class_emb = class_conditioner(class_labels)
    print(f"Class embeddings shape: {class_emb.shape}")
    
    # Test mask conditioner
    mask_conditioner = MaskConditioner()
    images = torch.randn(batch_size, 1, 32, 32, device=device)
    masks = mask_conditioner.create_random_mask(images.shape, mask_ratio=0.3)
    masked_images = mask_conditioner(images, masks)
    print(f"Masked images shape: {masked_images.shape}")
    
    # Test text conditioner
    text_conditioner = TextConditioner()
    text = ["hello", "world", "test", "diffusion"]
    text_emb = text_conditioner(text)
    print(f"Text embeddings shape: {text_emb.shape}")
    
    # Test multi conditioner
    multi_conditioner = MultiConditioner(
        class_conditioner=class_conditioner,
        mask_conditioner=mask_conditioner,
        text_conditioner=text_conditioner
    )
    cond_images, comb_emb = multi_conditioner(
        images=images,
        class_labels=class_labels,
        masks=masks,
        text=text
    )
    print(f"Combined embeddings shape: {comb_emb.shape}") 