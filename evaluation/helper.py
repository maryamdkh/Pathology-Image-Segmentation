"""
Functional evaluation orchestration for segmentation models.
"""

import torch
from typing import Dict, Any, List
from pathlib import Path
import json
import logging
import numpy as np

from training.metrics import iou_coeff, dice_coeff, precision_recall

logger = logging.getLogger(__name__)

def find_optimal_threshold_comprehensive(model, val_loader, device):
    """
    Find optimal threshold using multiple metrics
    """
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_metric = 0
    
    model.eval()
    
    results = []
    
    for thresh in thresholds:
        ious = []
        dices = []
        precisions = []
        recalls = []
        
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            with torch.no_grad():
                preds = torch.sigmoid(model(images))
                preds_bin = (preds > thresh).float()
                
                # Calculate multiple metrics
                iou = iou_coeff(preds_bin, masks)
                dice = dice_coeff(preds_bin, masks)
                precision, recall = precision_recall(preds_bin, masks)
                
                ious.append(iou.item())
                dices.append(dice.item())
                precisions.append(precision)
                recalls.append(recall)
        
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-8)
        
        # You can choose which metric to optimize
        current_metric = mean_dice   # or mean_f1, or mean_iou
        
        results.append({
            'threshold': thresh,
            'iou': mean_iou,
            'dice': mean_dice,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': mean_f1
        })
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = thresh
    
    # Print comprehensive results
    print(f"\nOptimal threshold: {best_threshold:.3f}")
    best_result = [r for r in results if r['threshold'] == best_threshold][0]
    print(f"IoU: {best_result['iou']:.4f}, Dice: {best_result['dice']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
    
    return best_threshold,best_result, results

def save_evaluation_results(
    evaluation_report: Dict[str, Any],
    save_dir: Path,
    experiment_name: str
) -> None:
    """
    Save evaluation results to structured directory.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics
    metrics_path = save_dir / f"{experiment_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Save summary
    summary_path = save_dir / f"{experiment_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(generate_evaluation_summary(evaluation_report))
    
    logger.info(f"Saved evaluation results to: {save_dir}")

def generate_evaluation_summary(evaluation_report: Dict[str, Any]) -> str:
    """Generate human-readable evaluation summary."""
    agg_metrics = evaluation_report["aggregate_metrics"]
    
    summary = [
        "Segmentation Model Evaluation Summary",
        "=" * 40,
        f"Total Images Evaluated: {evaluation_report['summary']['total_images']}",
        f"Threshold: {evaluation_report['summary']['threshold']}",
        "",
        "Aggregate Metrics:",
        f"  True Positive Rate: {agg_metrics['true_positive_rate']:.4f}",
        f"  True Negative Rate: {agg_metrics['true_negative_rate']:.4f}", 
        f"  Precision: {agg_metrics['precision']:.4f}",
        f"  Recall: {agg_metrics['recall']:.4f}",
        f"  F1 Score: {agg_metrics['f1_score']:.4f}",
        f"  IoU: {agg_metrics['iou']:.4f}",
        f"  Dice Coefficient: {agg_metrics['dice_coefficient']:.4f}",
        f"  Accuracy: {agg_metrics['accuracy']:.4f}",
    ]
    
    return "\n".join(summary)