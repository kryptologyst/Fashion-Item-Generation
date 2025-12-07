"""Evaluation metrics for fashion item generation."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from sklearn.metrics import precision_recall_curve
import lpips
from clean_fid import fid

from ..utils import get_device


class InceptionScore:
    """Calculate Inception Score for generated images."""
    
    def __init__(self, device: torch.device, batch_size: int = 64, splits: int = 10):
        """Initialize Inception Score calculator.
        
        Args:
            device: Device to run calculations on
            batch_size: Batch size for processing
            splits: Number of splits for calculation
        """
        self.device = device
        self.batch_size = batch_size
        self.splits = splits
        
        # Load pre-trained Inception model
        from torchvision.models import inception_v3
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
        # Remove final classification layer
        self.inception.fc = nn.Identity()
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using Inception model.
        
        Args:
            images: Input images
            
        Returns:
            Feature vectors
        """
        # Resize images to 299x299 for Inception
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features
    
    def calculate(self, generated_images: torch.Tensor) -> Tuple[float, float]:
        """Calculate Inception Score.
        
        Args:
            generated_images: Generated images tensor
            
        Returns:
            Tuple of (mean IS, std IS)
        """
        features = self.get_features(generated_images)
        
        # Calculate softmax probabilities
        probs = F.softmax(features, dim=1)
        
        # Calculate IS for each split
        split_size = len(probs) // self.splits
        scores = []
        
        for i in range(self.splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_probs = probs[start_idx:end_idx]
            
            # Calculate marginal distribution
            marginal = split_probs.mean(dim=0)
            
            # Calculate KL divergence
            kl_div = split_probs * (torch.log(split_probs + 1e-16) - torch.log(marginal + 1e-16))
            kl_div = kl_div.sum(dim=1)
            
            # Calculate IS for this split
            is_score = torch.exp(kl_div.mean())
            scores.append(is_score.item())
        
        return np.mean(scores), np.std(scores)


class PrecisionRecall:
    """Calculate Precision and Recall for generated images."""
    
    def __init__(self, device: torch.device, k: int = 3, batch_size: int = 64):
        """Initialize Precision/Recall calculator.
        
        Args:
            device: Device to run calculations on
            k: Number of nearest neighbors
            batch_size: Batch size for processing
        """
        self.device = device
        self.k = k
        self.batch_size = batch_size
    
    def calculate(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> Tuple[float, float]:
        """Calculate Precision and Recall.
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            
        Returns:
            Tuple of (precision, recall)
        """
        # Flatten images
        real_flat = real_images.view(real_images.size(0), -1)
        gen_flat = generated_images.view(generated_images.size(0), -1)
        
        # Calculate pairwise distances
        real_distances = torch.cdist(real_flat, real_flat)
        gen_distances = torch.cdist(gen_flat, gen_flat)
        cross_distances = torch.cdist(gen_flat, real_flat)
        
        # Calculate Precision: for each generated image, find k nearest real images
        gen_to_real_distances, _ = torch.topk(cross_distances, k=self.k, dim=1, largest=False)
        gen_to_gen_distances, _ = torch.topk(gen_distances, k=self.k, dim=1, largest=False)
        
        precision = (gen_to_real_distances.mean(dim=1) < gen_to_gen_distances.mean(dim=1)).float().mean()
        
        # Calculate Recall: for each real image, find k nearest generated images
        real_to_gen_distances, _ = torch.topk(cross_distances.t(), k=self.k, dim=1, largest=False)
        real_to_real_distances, _ = torch.topk(real_distances, k=self.k, dim=1, largest=False)
        
        recall = (real_to_gen_distances.mean(dim=1) < real_to_real_distances.mean(dim=1)).float().mean()
        
        return precision.item(), recall.item()


class LPIPSDiversity:
    """Calculate LPIPS diversity for generated images."""
    
    def __init__(self, device: torch.device, net: str = "alex", batch_size: int = 64):
        """Initialize LPIPS diversity calculator.
        
        Args:
            device: Device to run calculations on
            net: LPIPS network type ('alex', 'vgg', 'squeeze')
            batch_size: Batch size for processing
        """
        self.device = device
        self.batch_size = batch_size
        self.lpips_model = lpips.LPIPS(net=net).to(device)
    
    def calculate(self, generated_images: torch.Tensor) -> float:
        """Calculate LPIPS diversity.
        
        Args:
            generated_images: Generated images tensor
            
        Returns:
            Average LPIPS diversity
        """
        # Convert to [-1, 1] range for LPIPS
        images = generated_images * 2.0 - 1.0
        
        # Calculate pairwise LPIPS distances
        n_images = images.size(0)
        total_distance = 0.0
        count = 0
        
        for i in range(n_images):
            for j in range(i + 1, n_images):
                img1 = images[i:i+1]
                img2 = images[j:j+1]
                
                # Convert grayscale to RGB for LPIPS
                if img1.size(1) == 1:
                    img1 = img1.repeat(1, 3, 1, 1)
                    img2 = img2.repeat(1, 3, 1, 1)
                
                with torch.no_grad():
                    distance = self.lpips_model(img1, img2)
                    total_distance += distance.item()
                    count += 1
        
        return total_distance / count if count > 0 else 0.0


class EvaluationMetrics:
    """Comprehensive evaluation metrics for fashion generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation metrics.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = get_device({"auto": True})
        
        # Initialize metric calculators
        self.inception_score = InceptionScore(
            device=self.device,
            batch_size=config.get("inception_score", {}).get("batch_size", 64),
            splits=config.get("inception_score", {}).get("splits", 10),
        )
        
        self.precision_recall = PrecisionRecall(
            device=self.device,
            k=config.get("precision_recall", {}).get("k", 3),
            batch_size=config.get("precision_recall", {}).get("batch_size", 64),
        )
        
        self.lpips_diversity = LPIPSDiversity(
            device=self.device,
            net=config.get("lpips_diversity", {}).get("net", "alex"),
            batch_size=config.get("lpips_diversity", {}).get("batch_size", 64),
        )
    
    def calculate_fid(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """Calculate FID score.
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            
        Returns:
            FID score
        """
        # Convert tensors to numpy arrays
        real_np = real_images.cpu().numpy()
        gen_np = generated_images.cpu().numpy()
        
        # Reshape to (N, H, W, C) for clean-fid
        if real_np.ndim == 4 and real_np.shape[1] == 1:
            real_np = real_np.squeeze(1)  # Remove channel dimension
            gen_np = gen_np.squeeze(1)
        
        # Calculate FID
        fid_score = fid.compute_fid_from_arrays(real_np, gen_np)
        return fid_score
    
    def evaluate(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate generated images against real images.
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["fid", "inception_score", "precision_recall", "lpips_diversity"]
        
        results = {}
        
        # Move tensors to device
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)
        
        if "fid" in metrics:
            results["fid"] = self.calculate_fid(real_images, generated_images)
        
        if "inception_score" in metrics:
            is_mean, is_std = self.inception_score.calculate(generated_images)
            results["inception_score_mean"] = is_mean
            results["inception_score_std"] = is_std
        
        if "precision_recall" in metrics:
            precision, recall = self.precision_recall.calculate(real_images, generated_images)
            results["precision"] = precision
            results["recall"] = recall
        
        if "lpips_diversity" in metrics:
            results["lpips_diversity"] = self.lpips_diversity.calculate(generated_images)
        
        return results
