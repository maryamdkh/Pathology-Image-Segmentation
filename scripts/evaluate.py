#!/usr/bin/env python3
"""
Evaluation script for segmentation model.
Supports both single model and majority voting ensemble.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.utils import get_device, free_gpu_memory
from configs.config import load_config, setup_directories
from utils.logging import setup_mlflow_logger
from models.helper import build_seg_model
from data.helper import create_dataloaders
from evaluation.metrics_calculator import calculate_detailed_segmentation_report
from evaluation.visualization import generate_evaluation_visualizations
from evaluation.helper import find_optimal_threshold_comprehensive

def setup_logging(level=logging.INFO):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to model checkpoint (for single model mode)"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="evaluation",
        help="Experiment name for tracking"
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        default=None,
        help="MLflow run ID to log to (optional)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary prediction"
    )
    parser.add_argument(
        "--find-optimal-threshold",
        action="store_true",
        help="Find optimal threshold before evaluation"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization for sample images"
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of sample visualizations to generate"
    )
    
    return parser.parse_args()

def setup_evaluation(config: dict, experiment_name: str, run_id: Optional[str] = None):
    """Setup all components for evaluation."""
    setup_directories(config)
    device = get_device(prefer=config["training"].get("device", "auto"))
    logger = logging.getLogger(__name__)
    
    mlflow_logger = setup_mlflow_logger(
        experiment_name=experiment_name,
        config=config,
        run_id=run_id,
    )
    
    return device, mlflow_logger, logger

def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting evaluation process...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Split: {args.split}")
        
        # Setup evaluation environment
        device, mlflow_logger, logger = setup_evaluation(
            config, args.experiment, args.run_id
        )
        
        # Free GPU memory before starting
        free_gpu_memory()
        
        # Handle checkpoint for single model mode
        if args.checkpoint and config["model"]["inference_mode"] == "single":
            config['model']['single_model']['checkpoint_path'] = args.checkpoint
            logger.info(f"Using single model checkpoint: {args.checkpoint}")
        
        # Load model
        logger.info(f"Building model in {config['model']['inference_mode']} mode")
        model = build_seg_model(config)
        
        # Create dataloader
        logger.info(f"Creating dataloader for {args.split} split")        
        _, dataloaders = create_dataloaders(config, ["val","test"])

        # Find optimal threshold if requested
        if args.find_optimal_threshold:
            logger.info("🔍 Finding optimal threshold on the validation data...")
            optimal_threshold,best_result, threshold_results = find_optimal_threshold_comprehensive(
                model=model,
                val_loader=dataloaders["val"],
                device=device
            )
            args.threshold = optimal_threshold
            logger.info(f"Using optimal threshold: {optimal_threshold:.3f}")
            
            # Log threshold results to MLflow
            mlflow_logger.log_metrics({
                'optimal_threshold': optimal_threshold,
                'optimal_iou': best_result['iou'],  
                'optimal_dice': best_result['dice'],  
            })
        
        # Compute metrics
        logger.info(f"Computing metrics on {args.split} data...")
        evaluation_report = calculate_detailed_segmentation_report(
            model=model,
            dataloader=dataloaders[args.split],
            device=device,
            threshold=args.threshold
        )
        
        # Log metrics to MLflow
        mlflow_logger.log_metrics(evaluation_report["aggregate_metrics"])

        # Print summary to console
        logger.info("📊 Evaluation Results Summary:")
        for metric, value in evaluation_report["aggregate_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info(f"Generating {args.num_visualizations} sample visualizations...")
            generate_evaluation_visualizations(
                model=model,
                dataloader=dataloaders[args.split],
                device=device,
                output_dir=Path("evaluation_visualizations"),
                num_samples=args.num_visualizations,
                threshold=args.threshold
            )
        
        logger.info("Evaluation completed successfully!")
        
        # Cleanup
        mlflow_logger.finish()
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()