"""Utility functions for fashion item generation project."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_config: Dict[str, Any]) -> torch.device:
    """Get the best available device based on configuration.
    
    Args:
        device_config: Device configuration dictionary
        
    Returns:
        PyTorch device
    """
    if device_config.get("auto", True):
        if torch.cuda.is_available() and device_config.get("cuda", True):
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and device_config.get("mps", True):
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    # Manual device selection
    if device_config.get("cuda", False) and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_config.get("mps", False) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size as string (e.g., "1.2M", "45.6M")
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


def denormalize_tensor(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    return tensor * std + mean


def clamp_tensor(tensor: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """Clamp tensor values to specified range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped tensor
    """
    return torch.clamp(tensor, min_val, max_val)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """Initialize EMA.
        
        Args:
            model: Model to apply EMA to
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters.
        
        Args:
            model: Model to update EMA from
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Apply EMA parameters to model.
        
        Args:
            model: Model to apply EMA parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original parameters.
        
        Args:
            model: Model to restore parameters for
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
