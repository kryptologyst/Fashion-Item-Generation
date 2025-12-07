"""DCGAN implementation for fashion item generation."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(nn.Module):
    """Spectral normalization layer."""
    
    def __init__(self, module: nn.Module, name: str = "weight", n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12):
        """Initialize spectral normalization.
        
        Args:
            module: Module to apply spectral norm to
            name: Name of weight parameter
            n_power_iterations: Number of power iterations
            dim: Dimension to normalize
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        
        # Make a copy of the weight tensor
        weight = getattr(module, name)
        height = weight.size(dim)
        u = weight.new_empty(height).normal_(0, 1)
        v = weight.new_empty(weight.size(0)).normal_(0, 1)
        
        self.register_buffer("u", u)
        self.register_buffer("v", v)
    
    def _update_vectors(self) -> None:
        """Update u and v vectors using power iteration."""
        weight = getattr(self.module, self.name)
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight.view(weight.size(0), -1).t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.mv(weight.view(weight.size(0), -1), v), dim=0, eps=self.eps)
        
        self.u.copy_(u)
        self.v.copy_(v)
    
    def forward(self, *args):
        """Forward pass with spectral normalization."""
        self._update_vectors()
        weight = getattr(self.module, self.name)
        sigma = torch.dot(self.u, torch.mv(weight.view(weight.size(0), -1), self.v))
        weight = weight / sigma
        setattr(self.module, self.name, weight)
        return self.module(*args)


class Generator(nn.Module):
    """DCGAN Generator for fashion item generation."""
    
    def __init__(
        self,
        z_dim: int = 100,
        img_channels: int = 1,
        img_size: int = 28,
        hidden_dims: List[int] = [256, 512, 1024],
        use_spectral_norm: bool = True,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize Generator.
        
        Args:
            z_dim: Dimension of latent vector
            img_channels: Number of image channels
            img_size: Size of generated images
            hidden_dims: List of hidden layer dimensions
            use_spectral_norm: Whether to use spectral normalization
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Build layers
        layers = []
        in_dim = z_dim
        
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, img_channels * img_size * img_size))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    self.network[i] = SpectralNorm(layer)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from latent vectors.
        
        Args:
            z: Latent vectors of shape (batch_size, z_dim)
            
        Returns:
            Generated images of shape (batch_size, img_channels, img_size, img_size)
        """
        x = self.network(z)
        return x.view(-1, self.img_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    """DCGAN Discriminator for fashion item generation."""
    
    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        hidden_dims: List[int] = [64, 128],
        use_spectral_norm: bool = True,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize Discriminator.
        
        Args:
            img_channels: Number of image channels
            img_size: Size of input images
            hidden_dims: List of hidden layer dimensions
            use_spectral_norm: Whether to use spectral normalization
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate feature map size after convolutions
        feature_size = img_size // (2 ** len(hidden_dims))
        
        # Build convolutional layers
        conv_layers = []
        in_channels = img_channels
        
        for i, out_channels in enumerate(hidden_dims):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
            if use_batch_norm and i > 0:  # No batch norm on first layer
                conv_layers.append(nn.BatchNorm2d(out_channels))
            
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * feature_size * feature_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            for i, layer in enumerate(self.conv_layers):
                if isinstance(layer, nn.Conv2d):
                    self.conv_layers[i] = SpectralNorm(layer)
            
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    self.classifier[i] = SpectralNorm(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake.
        
        Args:
            x: Input images of shape (batch_size, img_channels, img_size, img_size)
            
        Returns:
            Classification scores of shape (batch_size, 1)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


class DCGAN(nn.Module):
    """DCGAN model combining Generator and Discriminator."""
    
    def __init__(
        self,
        z_dim: int = 100,
        img_channels: int = 1,
        img_size: int = 28,
        generator_config: Optional[dict] = None,
        discriminator_config: Optional[dict] = None,
    ):
        """Initialize DCGAN.
        
        Args:
            z_dim: Dimension of latent vector
            img_channels: Number of image channels
            img_size: Size of images
            generator_config: Generator configuration
            discriminator_config: Discriminator configuration
        """
        super().__init__()
        
        generator_config = generator_config or {}
        discriminator_config = discriminator_config or {}
        
        self.generator = Generator(
            z_dim=z_dim,
            img_channels=img_channels,
            img_size=img_size,
            **generator_config
        )
        
        self.discriminator = Discriminator(
            img_channels=img_channels,
            img_size=img_size,
            **discriminator_config
        )
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate images.
        
        Args:
            batch_size: Number of images to generate
            device: Device to generate on
            
        Returns:
            Generated images
        """
        z = torch.randn(batch_size, self.generator.z_dim, device=device)
        return self.generator(z)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator.
        
        Args:
            z: Latent vectors
            
        Returns:
            Generated images
        """
        return self.generator(z)
