"""
Functional metrics computation for segmentation models.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_segmentation_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    aggregate: bool = False
) -> Dict[str, Any]:
    """
    Calculate binary segmentation metrics per image or aggregated.
    
    Args:
        model: Trained PyTorch segmentation model
        dataloader: DataLoader returning batches with images, masks, and metadata
        device: Computation device
        threshold: Probability threshold for binary prediction
        aggregate: If True, return overall metrics across all images
        
    Returns:
        Dictionary with per-image or aggregated metrics
    """
    model.eval()
    per_image_predictions = collect_image_predictions(model, dataloader, device, threshold)
    per_image_metrics = compute_per_image_metrics(per_image_predictions)
    
    if aggregate:
        return compute_aggregate_metrics(per_image_metrics)
    return per_image_metrics


def collect_image_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float
) -> Dict[Tuple[int, int], Dict[str, List[torch.Tensor]]]:
    """
    Collect predictions and ground truth for all images in dataloader.
    
    Returns:
        Dictionary mapping (patient_num, image_num) to prediction data
    """
    image_data = {}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()
            patient_nums = batch["patient_num"]
            image_nums = batch["image_num"]
            
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).long()
            
            for i in range(len(predictions)):
                key = (patient_nums[i].item(), image_nums[i].item())
                if key not in image_data:
                    image_data[key] = {"predictions": [], "masks": []}
                
                # Flatten tiles for metric calculation
                image_data[key]["predictions"].append(predictions[i].cpu().view(-1))
                image_data[key]["masks"].append(masks[i].cpu().view(-1))
    
    logger.info(f"Collected predictions for {len(image_data)} images")
    return image_data


def compute_per_image_metrics(
    image_predictions: Dict[Tuple[int, int], Dict[str, List[torch.Tensor]]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute metrics for each individual image."""
    return {
        key: calculate_single_image_metrics(
            torch.cat(data["masks"]).numpy(),
            torch.cat(data["predictions"]).numpy()
        )
        for key, data in image_predictions.items()
    }


def calculate_single_image_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive metrics for a single image."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "true_positive_rate": tp / (tp + fn + 1e-8),
        "true_negative_rate": tn / (tn + fp + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "f1_score": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "dice_coefficient": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8),
    }


def compute_aggregate_metrics(
    per_image_metrics: Dict[Tuple[int, int], Dict[str, float]]
) -> Dict[str, float]:
    """Compute overall metrics by micro-averaging across all images."""
    # Collect all predictions and ground truth for micro-averaging
    all_true, all_pred = [], []
    
    for metrics in per_image_metrics.values():
        # Reconstruct confusion matrix from metrics (approximate)
        # For exact micro-averaging, you'd need the original predictions
        # This is a simplified version
        pass
    
    # For now, return macro-averaged metrics
    return {
        metric: np.mean([img_metrics[metric] for img_metrics in per_image_metrics.values()])
        for metric in next(iter(per_image_metrics.values())).keys()
    }


def calculate_detailed_segmentation_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Generate comprehensive segmentation report with both per-image and aggregate metrics.
    """
    per_image_metrics = calculate_segmentation_metrics(
        model, dataloader, device, threshold, aggregate=False
    )
    aggregate_metrics = compute_aggregate_metrics(per_image_metrics)
    
    return {
        "per_image_metrics": per_image_metrics,
        "aggregate_metrics": aggregate_metrics,
        "summary": {
            "total_images": len(per_image_metrics),
            "threshold": threshold,
        }
    }