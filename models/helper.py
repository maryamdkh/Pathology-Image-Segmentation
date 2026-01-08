import torch
import segmentation_models_pytorch as smp
from typing import Dict
from models.encoders.timm_encoder import TimmUniversalEncoder
from models.wrappers.UnetPlusPlus import UnetPlusPlus 
from models.wrappers.ClassificationModel import EncoderClassifier

def extract_unetpp_encoder_state_dict(unetpp_state: dict) -> dict:
    """
    Extract encoder-only weights from a Unet++ checkpoint.

    Expected key format:
        encoder.xxx

    Returns:
        dict: state_dict compatible with TimmUniversalEncoder
    """
    encoder_state = {}

    for key, value in unetpp_state.items():
        if key.startswith("encoder."):
            encoder_state[key.replace("encoder.", "", 1)] = value

    if not encoder_state:
        raise RuntimeError(
            "❌ No encoder weights found. "
            "This checkpoint does not look like a Unet++ model."
        )

    return encoder_state


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
        "TimmUnetPlusPlus": UnetPlusPlus

    }

    if architecture not in model_factory:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {list(model_factory.keys())}")
    
    if architecture == "TimmUnetPlusPlus":
        encoder = TimmUniversalEncoder(
            name=encoder_name,
            pretrained=True if encoder_weights else False
        )

        model = UnetPlusPlus(
            encoder=encoder,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            encoder_depth=model_cfg.get("encoder_depth", 5),
        )
    else:

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


def build_single_classifier_model(
    model_cfg: Dict,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build a classification model.

    Priority:
    1) Resume full classifier training from checkpoint_path
    2) Initialize encoder from Unet++ checkpoint (encoder_weights_path)
    3) Train from scratch
    """

    encoder_name = model_cfg.get("encoder_name", "")
    encoder_weights_path = model_cfg.get("encoder_weights_path", None)
    checkpoint_path = model_cfg.get("checkpoint_path", None)

    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("classes", 1)

    encoder_depth = model_cfg.get("encoder_depth", 5)
    output_stride = model_cfg.get("output_stride", 32)

    classifier_hidden_dim = model_cfg.get("classifier_hidden_dim", None)
    dropout = model_cfg.get("dropout", 0.3)

    # --- Build encoder ---
    encoder = TimmUniversalEncoder(
        name=encoder_name,
        pretrained=False,
        in_channels=in_channels,
        depth=encoder_depth,
        output_stride=output_stride,
    )

    # --- Build classifier ---
    model = EncoderClassifier(
        encoder=encoder,
        num_classes=num_classes,
        dropout=dropout,
        hidden_dim=classifier_hidden_dim,
    )

    # ==========================================================
    # 1️⃣ Resume classifier training (FULL model)
    # ==========================================================
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=True,
        )

        print(f"🔁 Resumed classifier training from checkpoint:")
        print(f"   {checkpoint_path}")
        return model.to(device)

    # ==========================================================
    # 2️⃣ Load encoder-only weights from Unet++
    # ==========================================================
    if encoder_weights_path is not None:
        checkpoint = torch.load(encoder_weights_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)

        encoder_state = extract_unetpp_encoder_state_dict(state_dict)

        missing, unexpected = model.encoder.load_state_dict(
            encoder_state,
            strict=False,
        )

        print(f"✅ Initialized encoder from Unet++ checkpoint:")
        print(f"   {encoder_weights_path}")

        if missing:
            print(f"⚠️ Missing encoder keys: {len(missing)}")
        if unexpected:
            print(f"⚠️ Unexpected encoder keys: {len(unexpected)}")

    # ==========================================================
    # 3️⃣ Fresh initialization
    # ==========================================================
    print("🆕 Classifier initialized from scratch (no pretrained weights)")

    return model.to(device)
