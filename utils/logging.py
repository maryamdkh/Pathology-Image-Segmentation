import mlflow
from pathlib import Path
from typing import Any, Dict,Optional
import logging
logger = logging.getLogger(__name__)


class MLflowLogger:
    def __init__(self, experiment_name: str, config: dict, run_name: str = None,tracking_dir: str = "./mlruns",
                  run_id: str = None):
                 
        """
        Initialize an MLflow logger with local storage.

        Args:
            experiment_name (str): Name of the experiment.
            config (dict): Full experiment config (parameters, model, loss, etc.).
            run_name (str, optional): Optional run name. Defaults to timestamp.
            tracking_dir (str): Local directory to store MLflow runs.
            run_id (str, optional): Existing run ID to attach to. Defaults to None.

        """
        # Ensure local tracking
        mlflow.set_tracking_uri(f"file:{tracking_dir}")
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if run_id is not None:
            # Attach to existing run
            self.run = mlflow.start_run(run_id=run_id, nested=False)
        else:
            self.run = mlflow.start_run(run_name=run_name)
            self.log_config(config)

        self.run_id = self.run.info.run_id


    def log_config(self, config: dict, prefix: str = ""):
        """
        Recursively log a dictionary as MLflow parameters.
        Nested dict keys are flattened with dot notation.
        """
        for k, v in config.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                self.log_config(v, prefix=key)
            else:
                # Convert lists or other types to string
                mlflow.log_param(key, str(v))

    def log_metrics(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name: str):
        """Save PyTorch model to MLflow as artifact."""
        mlflow.pytorch.log_model(model, model_name)

    def log_artifact(self, file_path: str):
        """Log additional files (plots, configs, visualizations)."""
        file_path = Path(file_path)
        if file_path.exists():
            mlflow.log_artifact(str(file_path))
        else:
            print(f"Artifact {file_path} does not exist, skipping.")

    def finish(self):
        """End MLflow run."""
        mlflow.end_run()



def setup_mlflow_logger(
    experiment_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tracking_dir: Optional[Path] = None,
    run_id: Optional[str] = None
) -> MLflowLogger:
    """
    Setup and configure MLflow logger.
    
    Args:
        experiment_name: Name of the MLflow experiment
        config: Configuration dictionary to log
        run_name: Specific run name (optional)
        tracking_dir: Path to MLflow tracking directory
        run_id: Existing run ID to resume (optional)
    
    Returns:
        MLflowLogger instance
    """
    try:
        logger.info("Setting up MLflow logger...")
        
        mlflow_logger = MLflowLogger(
            experiment_name=experiment_name,
            config=config,
            run_name=run_name,
            tracking_uri=tracking_dir,
            run_id=run_id
        )
        
        logger.info(f"✅ MLflow logger setup complete")
        logger.info(f"   Run ID: {mlflow_logger.run_id}")
        logger.info(f"   Experiment: {experiment_name}")
        
        return mlflow_logger
        
    except Exception as e:
        logger.error(f"❌ Failed to setup MLflow logger: {e}")
        raise

