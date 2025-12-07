#!/usr/bin/env python3
"""Main training script for fashion item generation."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from src.fashion_generation.training import GANTrainer
from src.fashion_generation.data import create_data_module
from src.fashion_generation.utils import set_seed, get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fashion item generation model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--overrides",
        nargs="*",
        help="Override configuration values (e.g., training.max_epochs=50)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on test set"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate samples after training"
    )
    
    return parser.parse_args()


def setup_logging(config: Dict[str, Any]) -> pl.loggers.Logger:
    """Setup logging configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured logger
    """
    if config.get("wandb", {}).get("enabled", False):
        return WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            save_dir=config.get("logs_dir", "logs"),
        )
    else:
        return TensorBoardLogger(
            save_dir=config.get("logs_dir", "logs"),
            name="fashion_generation",
        )


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Setup training callbacks.
    
    Args:
        config: Training configuration
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get("checkpoint_dir", "checkpoints"),
        filename=config.get("checkpoint", {}).get("filename", "epoch_{epoch:03d}-fid_{val_fid:.4f}"),
        monitor=config.get("monitor", "val_fid"),
        mode=config.get("mode", "min"),
        save_top_k=config.get("save_top_k", 3),
        save_last=config.get("checkpoint", {}).get("save_last", True),
        every_n_epochs=config.get("checkpoint", {}).get("save_every_n_epochs", 10),
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.get("early_stopping", {}).get("enabled", False):
        early_stop_callback = EarlyStopping(
            monitor=config.get("monitor", "val_fid"),
            mode=config.get("mode", "min"),
            patience=config.get("early_stopping", {}).get("patience", 10),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    return callbacks


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Apply overrides
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Create directories
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    os.makedirs(config.paths.assets_dir, exist_ok=True)
    
    # Setup device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data module
    data_module = create_data_module(config.data)
    print(f"Data module created with batch size: {config.data.batch_size}")
    
    # Create trainer
    trainer_module = GANTrainer(
        model_config=config.model,
        training_config=config.training,
        evaluation_config=config.evaluation,
    )
    
    # Print model info
    from src.fashion_generation.utils import count_parameters, get_model_size
    print(f"Generator parameters: {get_model_size(trainer_module.model.generator)}")
    print(f"Discriminator parameters: {get_model_size(trainer_module.model.discriminator)}")
    
    # Setup logging
    logger = setup_logging(config.logging)
    
    # Setup callbacks
    callbacks = setup_callbacks(config.training)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        devices=1,
        accelerator="auto",
        precision=config.training.stability.precision,
        gradient_clip_val=config.training.stability.gradient_clip_val,
        gradient_clip_algorithm=config.training.stability.gradient_clip_algorithm,
        accumulate_grad_batches=config.training.stability.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        log_every_n_steps=config.training.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=config.deterministic,
    )
    
    # Train model
    if not args.test:
        print("Starting training...")
        trainer.fit(trainer_module, data_module, ckpt_path=args.resume)
        print("Training completed!")
    
    # Test model
    if args.test:
        print("Running evaluation on test set...")
        trainer.test(trainer_module, data_module)
        print("Evaluation completed!")
    
    # Generate samples
    if args.sample:
        print("Generating samples...")
        from src.fashion_generation.sampling import FashionSampler
        
        sampler = FashionSampler(trainer_module.model, device, config.sampling)
        
        # Generate sample grid
        grid = sampler.generate_grid(
            grid_size=config.sampling.grid_size,
            seed=config.seed,
            save_path=os.path.join(config.paths.assets_dir, "sample_grid.png")
        )
        
        print(f"Samples saved to {config.paths.assets_dir}/sample_grid.png")


if __name__ == "__main__":
    main()
