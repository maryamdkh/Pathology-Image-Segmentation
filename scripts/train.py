#!/usr/bin/env python3
"""
Training script for segmentation model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from utils.utils import get_device, free_gpu_memory
from configs.config import load_config, setup_directories
from utils.logging import setup_mlflow_logger
from training.trainer import train_model

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up two levels from scripts/train.py
sys.path.insert(0, str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train segmentation model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to the main configuration file"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="baseline",
        help="Experiment name for tracking"
    )
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="Specific run name (optional)"
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        default=None,
        help="Specific run id (optional)"
    )
    
    return parser.parse_args()

def setup_training(config: Dict[str, Any], experiment_name: str, run_id: Optional[str] = None):
    """Setup all components for training."""
    
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
        run_id= run_id,
    )
    
    logger.info(f"Training setup complete:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Run: {run_id or 'fresh start'}")
    
    return device, mlflow_logger, logger


def main():
    """Main training function."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting training process...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Experiment: {args.experiment}")
        
        # Setup training environment
        device, mlflow_logger, logger = setup_training(
            config, 
            args.experiment, 
            args.run_id
        )
        
        # Free GPU memory before starting
        free_gpu_memory()
        
        # Run training
        logger.info("Starting model training...")
        
        best_checkpoint_path = train_model(
            config=config,
            logger=mlflow_logger,
            device=device,
            verbose=True
        )
        
        # Log final results
        mlflow_logger.log_artifact(best_checkpoint_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved at: {best_checkpoint_path}")
        
        # Cleanup
        mlflow_logger.finish()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()