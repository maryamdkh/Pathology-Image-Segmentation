"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    Load and fully resolve configuration with environment variables.
    """
    # Load config
    config = OmegaConf.load(config_path)
    
    config_resolved = OmegaConf.to_container(config, resolve=True)
    
    return OmegaConf.create(config_resolved)

def save_config(config: Dict[str, Any], save_path: Path):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def setup_directories(config: Dict[str, Any]) -> None:
    
    directories = [
        config['paths']['checkpoint_dir'], 
        config['paths']['log_dir'],
    ]
    
    for directory_path in directories:
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)