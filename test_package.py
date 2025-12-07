#!/usr/bin/env python3
"""Quick test script to verify the fashion generation package works correctly."""

import torch
from omegaconf import OmegaConf

from src.fashion_generation.models import DCGAN
from src.fashion_generation.data import create_data_module
from src.fashion_generation.sampling import FashionSampler
from src.fashion_generation.utils import set_seed, get_device


def test_basic_functionality():
    """Test basic functionality of the fashion generation package."""
    print("Testing Fashion Generation Package...")
    
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    print("✓ Configuration loaded")
    
    # Set seed and device
    set_seed(config.seed, config.deterministic)
    device = get_device(config.device)
    print(f"✓ Device: {device}")
    
    # Create model
    model = DCGAN(**config.model)
    model.to(device)
    print("✓ Model created and moved to device")
    
    # Test model forward pass
    batch_size = 4
    z = torch.randn(batch_size, config.model.z_dim, device=device)
    generated = model(z)
    assert generated.shape == (batch_size, 1, 28, 28)
    print("✓ Model forward pass works")
    
    # Test discriminator
    fake_pred = model.discriminator(generated)
    assert fake_pred.shape == (batch_size, 1)
    print("✓ Discriminator works")
    
    # Create data module
    data_module = create_data_module(config.data)
    data_module.prepare_data()
    data_module.setup("fit")
    print("✓ Data module created")
    
    # Test data loading
    train_loader = data_module.train_dataloader()
    batch, labels = next(iter(train_loader))
    assert batch.shape[0] == config.data.batch_size
    print("✓ Data loading works")
    
    # Create sampler
    sampler = FashionSampler(model, device, config.sampling)
    samples = sampler.generate_samples(num_samples=4, seed=42)
    assert samples.shape == (4, 1, 28, 28)
    print("✓ Sampling works")
    
    print("\n🎉 All tests passed! The package is working correctly.")


if __name__ == "__main__":
    test_basic_functionality()
