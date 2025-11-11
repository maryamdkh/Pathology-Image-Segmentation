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

def save_config(config: DictConfig, save_path: Path):
    """Save configuration using OmegaConf to preserve syntax."""
    OmegaConf.save(config, save_path)
    print(f"Configuration saved to: {save_path}")

def setup_directories(config: Dict[str, Any]) -> None:
    
    directories = [
        config['training']['checkpoint_dir'], 
        config['paths']['log_dir'],
    ]
    
    for directory_path in directories:
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)