"""Training module for fashion item generation."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

from ..models import DCGAN
from ..utils import EMA, get_device
from .losses import GANLoss


class GANTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for GAN models."""
    
    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
        evaluation_config: Optional[DictConfig] = None,
    ):
        """Initialize GAN trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            evaluation_config: Evaluation configuration
        """
        super().__init__()
        
        self.model_config = model_config
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        
        # Initialize model
        self.model = DCGAN(**model_config)
        
        # Initialize loss function
        self.gan_loss = GANLoss(
            generator_loss=training_config.loss.generator_loss,
            discriminator_loss=training_config.loss.discriminator_loss,
            label_smoothing=training_config.loss.label_smoothing,
        )
        
        # Initialize EMA if enabled
        self.ema_enabled = model_config.get("ema", {}).get("enabled", False)
        if self.ema_enabled:
            self.ema = EMA(self.model.generator, decay=model_config.ema.decay)
        
        # Initialize evaluation metrics if provided
        if evaluation_config:
            from ..evaluation import EvaluationMetrics
            self.evaluator = EvaluationMetrics(evaluation_config)
        
        # Training state
        self.current_epoch = 0
        self.global_step_count = 0
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for generator and discriminator.
        
        Returns:
            Tuple of (generator_optimizer, discriminator_optimizer)
        """
        # Generator optimizer
        gen_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            **self.training_config.optimizer.generator
        )
        
        # Discriminator optimizer
        disc_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            **self.training_config.optimizer.discriminator
        )
        
        # Configure schedulers if provided
        schedulers = []
        
        if "scheduler" in self.training_config:
            gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_optimizer,
                **self.training_config.scheduler.generator
            )
            disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                disc_optimizer,
                **self.training_config.scheduler.discriminator
            )
            
            schedulers = [
                {"scheduler": gen_scheduler, "interval": "epoch", "frequency": 1},
                {"scheduler": disc_scheduler, "interval": "epoch", "frequency": 1},
            ]
        
        return [gen_optimizer, disc_optimizer], schedulers
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step for GAN.
        
        Args:
            batch: Batch of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Training step output
        """
        real_images, _ = batch
        
        # Generate fake images
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, self.model.generator.z_dim, device=self.device)
        fake_images = self.model.generator(z)
        
        # Train discriminator
        d_loss = self._train_discriminator(real_images, fake_images)
        
        # Train generator
        g_loss = self._train_generator(fake_images)
        
        # Update EMA
        if self.ema_enabled:
            self.ema.update(self.model.generator)
        
        # Log losses
        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": d_loss + g_loss, "d_loss": d_loss, "g_loss": g_loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step for GAN.
        
        Args:
            batch: Batch of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Validation step output
        """
        real_images, _ = batch
        
        # Generate fake images
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, self.model.generator.z_dim, device=self.device)
        fake_images = self.model.generator(z)
        
        # Calculate losses
        d_loss = self._calculate_discriminator_loss(real_images, fake_images)
        g_loss = self._calculate_generator_loss(fake_images)
        
        # Log losses
        self.log("val/d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if hasattr(self, "evaluator") and self.current_epoch % self.training_config.get("eval_every_n_epochs", 5) == 0:
            self._evaluate_model()
    
    def _train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        """Train discriminator.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            Discriminator loss
        """
        # Get optimizers
        gen_optimizer, disc_optimizer = self.optimizers()
        
        # Train discriminator
        disc_optimizer.zero_grad()
        
        d_loss = self._calculate_discriminator_loss(real_images, fake_images.detach())
        
        self.manual_backward(d_loss)
        disc_optimizer.step()
        
        return d_loss
    
    def _train_generator(self, fake_images: torch.Tensor) -> torch.Tensor:
        """Train generator.
        
        Args:
            fake_images: Generated images
            
        Returns:
            Generator loss
        """
        # Get optimizers
        gen_optimizer, disc_optimizer = self.optimizers()
        
        # Train generator
        gen_optimizer.zero_grad()
        
        g_loss = self._calculate_generator_loss(fake_images)
        
        self.manual_backward(g_loss)
        gen_optimizer.step()
        
        return g_loss
    
    def _calculate_discriminator_loss(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator loss.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            Discriminator loss
        """
        # Real images
        real_pred = self.model.discriminator(real_images)
        real_loss = self.gan_loss.discriminator_loss(real_pred, True)
        
        # Fake images
        fake_pred = self.model.discriminator(fake_images)
        fake_loss = self.gan_loss.discriminator_loss(fake_pred, False)
        
        return (real_loss + fake_loss) * 0.5
    
    def _calculate_generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss.
        
        Args:
            fake_images: Generated images
            
        Returns:
            Generator loss
        """
        fake_pred = self.model.discriminator(fake_images)
        return self.gan_loss.generator_loss(fake_pred, True)
    
    def _evaluate_model(self) -> None:
        """Evaluate model using configured metrics."""
        if not hasattr(self, "evaluator"):
            return
        
        # Generate samples for evaluation
        num_samples = self.training_config.get("eval_samples", 1000)
        z = torch.randn(num_samples, self.model.generator.z_dim, device=self.device)
        
        with torch.no_grad():
            generated_images = self.model.generator(z)
        
        # Get real images for comparison
        dataloader = self.trainer.datamodule.val_dataloader()
        real_images = []
        
        for batch, _ in dataloader:
            real_images.append(batch)
            if len(real_images) * batch.size(0) >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples].to(self.device)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate(real_images, generated_images)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_epoch=True, prog_bar=True)
    
    def generate_samples(self, num_samples: int = 64) -> torch.Tensor:
        """Generate samples using the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated images
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.generator.z_dim, device=self.device)
            
            if self.ema_enabled:
                # Use EMA parameters for generation
                self.ema.apply_shadow(self.model.generator)
                generated_images = self.model.generator(z)
                self.ema.restore(self.model.generator)
            else:
                generated_images = self.model.generator(z)
        
        return generated_images
