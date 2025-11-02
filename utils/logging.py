import mlflow
from pathlib import Path
from typing import Any, Dict,Optional
import logging
from datetime import datetime
from mlflow.tracking import MlflowClient

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


def explore_mlflow_runs(mlruns_path, experiment_name=None):
    """
    Explore and list all MLflow runs from a specified MLflow tracking directory.
    
    Args:
        mlruns_path (str): Path to the MLflow runs directory (e.g., "mlruns", "/path/to/mlruns")
        experiment_name (str, optional): Specific experiment name to filter by. Defaults to None.
    """
    # Set the tracking URI to the provided mlruns path
    mlflow.set_tracking_uri(mlruns_path)
    
    client = MlflowClient()

    # Get experiments
    experiments = client.search_experiments()
    print("🔍 Available Experiments:")
    print("=" * 80)

    for exp in experiments:
        # If experiment_name is specified, skip other experiments
        if experiment_name and exp.name != experiment_name:
            continue
            
        print(f"📂 Experiment: {exp.name} (ID: {exp.experiment_id})")

        # Get runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])

        if runs:
            print(f"   📊 Runs in '{exp.name}':")
            for run in runs:
                status = run.info.status
                start_time = datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else "N/A"

                print(f"      🆔 Run ID: {run.info.run_id}")
                print(f"         Name: {run.data.tags.get('mlflow.runName', 'Unnamed')}")
                print(f"         Status: {status}")
                print(f"         Start Time: {start_time}")
                print(f"         Metrics: {list(run.data.metrics.keys())[:3]}...")  # Show first 3 metrics
                print(f"         Parameters: {list(run.data.params.keys())[:3]}...")  # Show first 3 params
                print("         " + "-" * 50)
        else:
            print(f"   No runs in this experiment")
        print()

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
            run_id=run_id
        )
        
        logger.info(f"✅ MLflow logger setup complete")
        logger.info(f"   Run ID: {mlflow_logger.run_id}")
        logger.info(f"   Experiment: {experiment_name}")
        
        return mlflow_logger
        
    except Exception as e:
        logger.error(f"❌ Failed to setup MLflow logger: {e}")
        raise

