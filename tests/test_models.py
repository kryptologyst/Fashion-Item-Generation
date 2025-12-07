"""Unit tests for fashion generation models."""

import pytest
import torch
import torch.nn as nn

from src.fashion_generation.models import DCGAN, Generator, Discriminator, SpectralNorm
from src.fashion_generation.training import GANLoss
from src.fashion_generation.utils import EMA


class TestGenerator:
    """Test cases for Generator model."""
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = Generator(z_dim=100, img_channels=1, img_size=28)
        
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 1, 28, 28)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_with_spectral_norm(self):
        """Test generator with spectral normalization."""
        generator = Generator(
            z_dim=100,
            img_channels=1,
            img_size=28,
            use_spectral_norm=True
        )
        
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 1, 28, 28)
    
    def test_generator_different_sizes(self):
        """Test generator with different image sizes."""
        for img_size in [32, 64, 128]:
            generator = Generator(z_dim=100, img_channels=3, img_size=img_size)
            
            batch_size = 8
            z = torch.randn(batch_size, 100)
            
            output = generator(z)
            
            assert output.shape == (batch_size, 3, img_size, img_size)


class TestDiscriminator:
    """Test cases for Discriminator model."""
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = Discriminator(img_channels=1, img_size=28)
        
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_discriminator_with_spectral_norm(self):
        """Test discriminator with spectral normalization."""
        discriminator = Discriminator(
            img_channels=1,
            img_size=28,
            use_spectral_norm=True
        )
        
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_discriminator_different_sizes(self):
        """Test discriminator with different image sizes."""
        for img_size in [32, 64, 128]:
            discriminator = Discriminator(img_channels=3, img_size=img_size)
            
            batch_size = 8
            x = torch.randn(batch_size, 3, img_size, img_size)
            
            output = discriminator(x)
            
            assert output.shape == (batch_size, 1)


class TestDCGAN:
    """Test cases for DCGAN model."""
    
    def test_dcgan_forward(self):
        """Test DCGAN forward pass."""
        model = DCGAN(z_dim=100, img_channels=1, img_size=28)
        
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        generated = model(z)
        
        assert generated.shape == (batch_size, 1, 28, 28)
    
    def test_dcgan_generate(self):
        """Test DCGAN generate method."""
        model = DCGAN(z_dim=100, img_channels=1, img_size=28)
        device = torch.device("cpu")
        
        batch_size = 16
        generated = model.generate(batch_size, device)
        
        assert generated.shape == (batch_size, 1, 28, 28)
    
    def test_dcgan_end_to_end(self):
        """Test DCGAN end-to-end generation and discrimination."""
        model = DCGAN(z_dim=100, img_channels=1, img_size=28)
        
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        # Generate images
        generated = model(z)
        
        # Discriminate generated images
        fake_pred = model.discriminator(generated)
        
        assert fake_pred.shape == (batch_size, 1)
        assert fake_pred.min() >= 0.0
        assert fake_pred.max() <= 1.0


class TestSpectralNorm:
    """Test cases for SpectralNorm layer."""
    
    def test_spectral_norm_forward(self):
        """Test spectral normalization forward pass."""
        linear = nn.Linear(10, 5)
        spectral_linear = SpectralNorm(linear)
        
        x = torch.randn(16, 10)
        output = spectral_linear(x)
        
        assert output.shape == (16, 5)
    
    def test_spectral_norm_conv2d(self):
        """Test spectral normalization on Conv2d layer."""
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        spectral_conv = SpectralNorm(conv)
        
        x = torch.randn(16, 3, 32, 32)
        output = spectral_conv(x)
        
        assert output.shape == (16, 64, 32, 32)


class TestGANLoss:
    """Test cases for GAN loss functions."""
    
    def test_bce_loss(self):
        """Test BCE loss."""
        loss_fn = GANLoss(generator_loss="bce", discriminator_loss="bce")
        
        batch_size = 16
        pred = torch.rand(batch_size, 1)
        
        # Generator loss (target is real)
        g_loss = loss_fn.generator_loss(pred, True)
        assert g_loss.item() >= 0
        
        # Discriminator loss (target is fake)
        d_loss = loss_fn.discriminator_loss(pred, False)
        assert d_loss.item() >= 0
    
    def test_hinge_loss(self):
        """Test hinge loss."""
        loss_fn = GANLoss(generator_loss="hinge", discriminator_loss="hinge")
        
        batch_size = 16
        pred = torch.randn(batch_size, 1)
        
        # Generator loss
        g_loss = loss_fn.generator_loss(pred, True)
        assert g_loss.item() >= 0
        
        # Discriminator loss
        d_loss = loss_fn.discriminator_loss(pred, False)
        assert d_loss.item() >= 0
    
    def test_wasserstein_loss(self):
        """Test Wasserstein loss."""
        loss_fn = GANLoss(generator_loss="wasserstein", discriminator_loss="wasserstein")
        
        batch_size = 16
        pred = torch.randn(batch_size, 1)
        
        # Generator loss
        g_loss = loss_fn.generator_loss(pred, True)
        
        # Discriminator loss
        d_loss = loss_fn.discriminator_loss(pred, False)
        
        # Wasserstein loss can be negative
        assert isinstance(g_loss.item(), float)
        assert isinstance(d_loss.item(), float)


class TestEMA:
    """Test cases for EMA utility."""
    
    def test_ema_update(self):
        """Test EMA update."""
        model = Generator(z_dim=100, img_channels=1, img_size=28)
        ema = EMA(model, decay=0.999)
        
        # Update EMA
        ema.update(model)
        
        # Check that shadow parameters exist
        assert len(ema.shadow) > 0
        
        # Check that shadow parameters have correct shapes
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert ema.shadow[name].shape == param.shape
    
    def test_ema_apply_restore(self):
        """Test EMA apply and restore."""
        model = Generator(z_dim=100, img_channels=1, img_size=28)
        ema = EMA(model, decay=0.999)
        
        # Store original parameters
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # Update EMA
        ema.update(model)
        
        # Apply EMA
        ema.apply_shadow(model)
        
        # Check that parameters changed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.equal(param.data, original_params[name])
        
        # Restore original parameters
        ema.restore(model)
        
        # Check that parameters are restored
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(param.data, original_params[name])


if __name__ == "__main__":
    pytest.main([__file__])
