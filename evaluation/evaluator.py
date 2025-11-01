"""
Functional evaluation orchestration for segmentation models.
"""

import torch
from typing import Dict, Any, List
from pathlib import Path
import json
import logging

from evaluation.metrics_calculator import calculate_detailed_segmentation_report
from evaluation.visualizer import visualize_prediction_comparison, prepare_image_for_visualization, prepare_mask_for_visualization

logger = logging.getLogger(__name__)


def evaluate_model_performance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation returning detailed performance report.
    """
    logger.info("Starting comprehensive model evaluation...")
    
    evaluation_report = calculate_detailed_segmentation_report(
        model, dataloader, device, threshold
    )
    
    logger.info("✅ Model evaluation completed")
    return evaluation_report


def generate_evaluation_visualizations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
    tile_size: int = 512,
    overlap: int = 64
) -> List[Path]:
    """
    Generate sample visualizations for model evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    samples_processed = 0
    
    model.eval()
    
    for batch in dataloader:
        if samples_processed >= num_samples:
            break
            
        images = batch["image"]
        masks = batch["mask"]
        patient_nums = batch["patient_num"]
        image_nums = batch["image_num"]
        
        for i in range(len(images)):
            if samples_processed >= num_samples:
                break
                
            # Prepare image and mask
            image_np = prepare_image_for_visualization(images[i].numpy())
            mask_np = prepare_mask_for_visualization(masks[i].numpy(), image_np.shape[:2])
            
            save_path = output_dir / f"patient_{patient_nums[i].item()}_image_{image_nums[i].item()}.png"
            
            try:
                visualize_prediction_comparison(
                    model=model,
                    image=image_np,
                    device=device,
                    ground_truth_mask=mask_np,
                    tile_size=tile_size,
                    overlap=overlap,
                    save_path=save_path
                )
                
                saved_paths.append(save_path)
                samples_processed += 1
                logger.info(f"✅ Generated visualization {samples_processed}/{num_samples}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate visualization for sample {i}: {e}")
                continue
    
    logger.info(f"Generated {len(saved_paths)} visualization samples")
    return saved_paths


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