#!/usr/bin/env python3
"""Sampling script for fashion item generation."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from src.fashion_generation.sampling import FashionSampler, load_model_for_sampling
from src.fashion_generation.utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate fashion samples")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--grid_size",
        type=int,
        default=8,
        help="Size of the sample grid"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/generated",
        help="Output directory for generated samples"
    )
    
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate interpolation between two random samples"
    )
    
    parser.add_argument(
        "--interpolation_steps",
        type=int,
        default=10,
        help="Number of interpolation steps"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main sampling function."""
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, device = load_model_for_sampling(
        args.checkpoint,
        config.model,
        device
    )
    print("Model loaded successfully!")
    
    # Create sampler
    sampler = FashionSampler(model, device, config.sampling)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    if args.interpolate:
        print("Generating interpolation...")
        
        # Generate two random latent vectors
        z1 = torch.randn(1, model.generator.z_dim, device=device)
        z2 = torch.randn(1, model.generator.z_dim, device=device)
        
        # Interpolate
        interpolated = sampler.interpolate(
            z1, z2,
            num_steps=args.interpolation_steps,
            save_path=os.path.join(args.output_dir, "interpolation.png")
        )
        
        print(f"Interpolation saved to {args.output_dir}/interpolation.png")
    
    else:
        print(f"Generating {args.num_samples} samples...")
        
        # Generate sample grid
        grid = sampler.generate_grid(
            grid_size=args.grid_size,
            seed=args.seed,
            save_path=os.path.join(args.output_dir, "sample_grid.png")
        )
        
        print(f"Sample grid saved to {args.output_dir}/sample_grid.png")
        
        # Generate individual samples
        samples = sampler.generate_samples(
            num_samples=args.num_samples,
            seed=args.seed,
            save_path=os.path.join(args.output_dir, "samples.png")
        )
        
        print(f"Individual samples saved to {args.output_dir}/samples.png")
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
