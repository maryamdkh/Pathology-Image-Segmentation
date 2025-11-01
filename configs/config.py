"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: Path):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def setup_directories(config: Dict[str, Any]) -> None:
    
    directories = [
        config['paths']['checkpoint_dir'], 
        config['paths']['log_dir'],
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)