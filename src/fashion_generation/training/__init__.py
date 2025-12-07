"""Training package initialization."""

from .trainer import GANTrainer
from .losses import GANLoss, gradient_penalty

__all__ = ["GANTrainer", "GANLoss", "gradient_penalty"]
