"""Sampling utilities for fashion item generation."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from ..models import DCGAN
from ..utils import get_device, denormalize_tensor, clamp_tensor


class FashionSampler:
    """Sampling utilities for fashion generation models."""
    
    def __init__(
        self,
        model: DCGAN,
        device: torch.device,
        config: Optional[DictConfig] = None,
    ):
        """Initialize fashion sampler.
        
        Args:
            model: Trained DCGAN model
            device: Device to run sampling on
            config: Sampling configuration
        """
        self.model = model
        self.device = device
        self.config = config or {}
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_samples(
        self,
        num_samples: int = 64,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate fashion samples.
        
        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            save_path: Path to save generated images
            
        Returns:
            Generated images tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.generator.z_dim, device=self.device)
            generated_images = self.model.generator(z)
        
        # Denormalize for visualization
        generated_images = denormalize_tensor(generated_images)
        generated_images = clamp_tensor(generated_images, 0.0, 1.0)
        
        if save_path:
            self._save_samples(generated_images, save_path)
        
        return generated_images
    
    def generate_grid(
        self,
        grid_size: int = 8,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate a grid of fashion samples.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            seed: Random seed for reproducibility
            save_path: Path to save the grid image
            
        Returns:
            Grid image tensor
        """
        num_samples = grid_size * grid_size
        samples = self.generate_samples(num_samples, seed)
        
        # Create grid
        grid = vutils.make_grid(samples, nrow=grid_size, padding=2, normalize=True)
        
        if save_path:
            vutils.save_image(grid, save_path, normalize=True)
        
        return grid
    
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Interpolate between two latent vectors.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps
            save_path: Path to save interpolation images
            
        Returns:
            Interpolated images tensor
        """
        # Ensure latent vectors are on correct device
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        
        interpolated_images = []
        
        with torch.no_grad():
            for alpha in alphas:
                # Linear interpolation in latent space
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = self.model.generator(z_interp)
                
                # Denormalize
                img = denormalize_tensor(img)
                img = clamp_tensor(img, 0.0, 1.0)
                
                interpolated_images.append(img)
        
        # Stack images
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
        if save_path:
            grid = vutils.make_grid(interpolated_images, nrow=num_steps, padding=2, normalize=True)
            vutils.save_image(grid, save_path, normalize=True)
        
        return interpolated_images
    
    def generate_class_samples(
        self,
        class_indices: List[int],
        samples_per_class: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """Generate samples for specific fashion classes.
        
        Args:
            class_indices: List of class indices to generate
            samples_per_class: Number of samples per class
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping class indices to generated samples
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        class_samples = {}
        
        with torch.no_grad():
            for class_idx in class_indices:
                # Generate random latent vectors
                z = torch.randn(samples_per_class, self.model.generator.z_dim, device=self.device)
                
                # Generate images
                images = self.model.generator(z)
                
                # Denormalize
                images = denormalize_tensor(images)
                images = clamp_tensor(images, 0.0, 1.0)
                
                class_samples[class_idx] = images
        
        return class_samples
    
    def _save_samples(self, images: torch.Tensor, save_path: str) -> None:
        """Save generated samples.
        
        Args:
            images: Generated images tensor
            save_path: Path to save images
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create grid and save
        grid = vutils.make_grid(images, nrow=8, padding=2, normalize=True)
        vutils.save_image(grid, save_path, normalize=True)
    
    def visualize_samples(
        self,
        images: torch.Tensor,
        title: str = "Generated Fashion Items",
        figsize: Tuple[int, int] = (12, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize generated samples.
        
        Args:
            images: Generated images tensor
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Convert to numpy for matplotlib
        if images.dim() == 4:
            # Take first image if batch
            img = images[0].cpu().numpy()
        else:
            img = images.cpu().numpy()
        
        # Remove channel dimension if grayscale
        if img.shape[0] == 1:
            img = img.squeeze(0)
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()


def load_model_for_sampling(
    checkpoint_path: str,
    model_config: DictConfig,
    device: Optional[torch.device] = None,
) -> Tuple[DCGAN, torch.device]:
    """Load a trained model for sampling.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration
        device: Device to load model on
        
    Returns:
        Tuple of (loaded model, device)
    """
    if device is None:
        device = get_device({"auto": True})
    
    # Load model
    model = DCGAN(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoint
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Direct model checkpoint
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device
