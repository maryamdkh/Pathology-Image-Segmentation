#!/usr/bin/env python3
"""
Evaluation script for segmentation model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.utils import get_device, free_gpu_memory
from configs.config import load_config, setup_directories
from utils.logging import setup_mlflow_logger
from models.helper import build_seg_model
from data.helper import create_dataloaders
from evaluation.evaluator import evaluate_model_performance,generate_evaluation_visualizations


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
        required=True,
        help="Path to model checkpoint to evaluate"
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
    parser.add_argument(
        "--result-dir",
        type=str,
        default="evaluation_visualizations",
        help="Directory to save the results"
    )
    
    return parser.parse_args()


def setup_evaluation(config: Dict[str, Any], experiment_name: str, run_id: Optional[str] = None):
    """Setup all components for evaluation."""
    
    # Setup directories
    setup_directories(config)
    
    # Setup device
    device = get_device(prefer=config["training"].get("device", "auto"))
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Setup MLflow
    mlflow_logger = setup_mlflow_logger(
        experiment_name=experiment_name,
        config=config,
        run_id=run_id,
    )
    
    logger.info(f"Evaluation setup complete:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Run: {run_id or 'fresh start'}")
    
    return device, mlflow_logger, logger



def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger = logging.getLogger(__name__)
        results_dir = Path(args.result_dir)
        
        logger.info("Starting evaluation process...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Split: {args.split}")
        
        # Setup evaluation environment
        device, mlflow_logger, logger = setup_evaluation(
            config, 
            args.experiment, 
            args.run_id
        )
        
        # Free GPU memory before starting
        free_gpu_memory()
        
        # Load model
        logger.info(f"Loading model from: {args.checkpoint}")
        config['model']['checkpoint_path'] = args.checkpoint
        model = build_seg_model(config)
        
        # Create dataloader
        logger.info(f"Creating dataloader for {args.split} split")
        _,dataloaders = create_dataloaders(config,[args.split])
       
        
        # Compute metrics
        logger.info(f"Computing metrics on {args.split} data ...")
        evaluation_report = evaluate_model_performance(
            model=model,
            dataloader=dataloaders[args.split],
            device=device,
            threshold=args.threshold
        )
        
        # Log metrics to MLflow
        mlflow_logger.log_metrics(evaluation_report)

        # Print summary to console
        logger.info("📊 Evaluation Results Summary:")
        for metric, value in evaluation_report["aggregate_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info(f"Generating {args.num_visualizations} sample visualizations...")
            
            visualization_paths = generate_evaluation_visualizations(
                model=model,
                dataloader=dataloaders[args.split],
                device=device,
                output_dir=results_dir,
                num_samples=args.num_visualizations,
                tile_size=config["dataset"].get("tile_size", 512),
                overlap=config["dataset"].get("overlap", 64)
            )
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {results_dir}")
        
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