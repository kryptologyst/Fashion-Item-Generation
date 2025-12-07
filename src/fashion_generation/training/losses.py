"""Loss functions for GAN training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss:
    """GAN loss functions with different variants."""
    
    def __init__(
        self,
        generator_loss: str = "bce",
        discriminator_loss: str = "bce",
        label_smoothing: float = 0.0,
    ):
        """Initialize GAN loss.
        
        Args:
            generator_loss: Generator loss type ('bce', 'hinge', 'wasserstein')
            discriminator_loss: Discriminator loss type ('bce', 'hinge', 'wasserstein')
            label_smoothing: Label smoothing factor
        """
        self.generator_loss_type = generator_loss
        self.discriminator_loss_type = discriminator_loss
        self.label_smoothing = label_smoothing
        
        # Initialize loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def generator_loss(self, fake_pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate generator loss.
        
        Args:
            fake_pred: Discriminator predictions on fake images
            target_is_real: Whether target is real (True) or fake (False)
            
        Returns:
            Generator loss
        """
        if self.generator_loss_type == "bce":
            return self._bce_loss(fake_pred, target_is_real)
        elif self.generator_loss_type == "hinge":
            return self._hinge_loss(fake_pred, target_is_real)
        elif self.generator_loss_type == "wasserstein":
            return self._wasserstein_loss(fake_pred, target_is_real)
        else:
            raise ValueError(f"Unknown generator loss type: {self.generator_loss_type}")
    
    def discriminator_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate discriminator loss.
        
        Args:
            pred: Discriminator predictions
            target_is_real: Whether target is real (True) or fake (False)
            
        Returns:
            Discriminator loss
        """
        if self.discriminator_loss_type == "bce":
            return self._bce_loss(pred, target_is_real)
        elif self.discriminator_loss_type == "hinge":
            return self._hinge_loss(pred, target_is_real)
        elif self.discriminator_loss_type == "wasserstein":
            return self._wasserstein_loss(pred, target_is_real)
        else:
            raise ValueError(f"Unknown discriminator loss type: {self.discriminator_loss_type}")
    
    def _bce_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Binary cross-entropy loss with optional label smoothing.
        
        Args:
            pred: Predictions
            target_is_real: Whether target is real
            
        Returns:
            BCE loss
        """
        if target_is_real:
            target = torch.ones_like(pred)
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            target = torch.zeros_like(pred)
            if self.label_smoothing > 0:
                target = target + 0.5 * self.label_smoothing
        
        return self.bce_loss(pred, target)
    
    def _hinge_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Hinge loss for GAN training.
        
        Args:
            pred: Predictions
            target_is_real: Whether target is real
            
        Returns:
            Hinge loss
        """
        if target_is_real:
            return F.relu(1 - pred).mean()
        else:
            return F.relu(1 + pred).mean()
    
    def _wasserstein_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Wasserstein loss for GAN training.
        
        Args:
            pred: Predictions
            target_is_real: Whether target is real
            
        Returns:
            Wasserstein loss
        """
        if target_is_real:
            return -pred.mean()
        else:
            return pred.mean()


def gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator model
        real_images: Real images
        fake_images: Generated images
        device: Device to run on
        lambda_gp: Gradient penalty weight
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_images.size(0)
    
    # Random interpolation between real and fake images
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated images
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    
    return penalty
