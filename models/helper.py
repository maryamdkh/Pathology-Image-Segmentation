import torch
import segmentation_models_pytorch as smp
from typing import Dict
from models.encoders.register_timm import register_common_transformers
register_common_transformers()

def build_seg_model(config: dict, device: torch.device = "cuda"):
    """
    Build a segmentation model based on config.
    Supports both single model and majority voting ensemble.
    
    Args:
        config (dict): Configuration containing model parameters
        device (torch.device): Device to map model weights to
        
    Returns:
        torch.nn.Module: The constructed model (single or ensemble)
    """
    model_cfg = config.get("model", {})
    inference_mode = model_cfg.get("inference_mode", "single")
    
    if inference_mode == "single":
        return _build_single_model(model_cfg["single_model"], device)
    elif inference_mode == "majority_vote":
        return _build_majority_vote_ensemble(model_cfg["majority_vote"], device)
    else:
        raise ValueError(f"Unknown inference_mode: {inference_mode}. Supported: 'single', 'majority_vote'")

def _build_single_model(model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Build a single segmentation model."""
    architecture = model_cfg.get("architecture", "UnetPlusPlus")
    encoder_name = model_cfg.get("encoder_name", "")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("classes", 1)
    checkpoint_path = model_cfg.get("checkpoint_path", None)

    model_factory = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "PAN": smp.PAN,
        "Linknet": smp.Linknet,
    }

    if architecture not in model_factory:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {list(model_factory.keys())}")

    model = model_factory[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # return raw logits
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Loaded pretrained model from: {checkpoint_path}")

    print(f"The {model.name} model has been built!!")

    return model.to(device)

def _build_majority_vote_ensemble(ensemble_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Build majority voting ensemble from multiple models."""
    from models.majority_vote import MajorityVotingEnsemble 
    
    model_configs = []
    for model_cfg in ensemble_cfg["models"]:
        model_configs.append({
            "model": {
                "architecture": model_cfg["architecture"],
                "encoder_name": model_cfg["encoder_name"],
                "encoder_weights": model_cfg["encoder_weights"],
                "in_channels": model_cfg["in_channels"],
                "classes": model_cfg["classes"],
                "checkpoint_path": model_cfg["checkpoint_path"]
            }
        })
    
    # Get optional parameters
    voting_strategy = ensemble_cfg.get("voting_strategy", "majority")
    confidence_threshold = ensemble_cfg.get("confidence_threshold", 0.5)
    
    ensemble = MajorityVotingEnsemble(
        model_configs=model_configs,
        device=device,
        voting_strategy=voting_strategy,
        confidence_threshold=confidence_threshold
    )
    
    print(f"✅ Built majority voting ensemble with {len(model_configs)} models")
    for model_cfg in ensemble_cfg["models"]:
        print(f"   - {model_cfg['name']} (trained on {model_cfg['trained_on']})")
    
    return ensemble
