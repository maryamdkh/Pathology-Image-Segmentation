import torch
import segmentation_models_pytorch as smp


def build_seg_model(config: dict, device: torch.device = "cuda"):
    """
    Build a segmentation model based on a config dictionary.

    Args:
        config (dict): Configuration containing model parameters.
        device (torch.device): Device to map model weights to.

    Returns:
        torch.nn.Module: The constructed model.
    """
    model_cfg = config.get("model", {})

    architecture = model_cfg.get("architecture", "UnetPlusPlus")
    encoder_name = model_cfg.get("encoder_name", "resnet50")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("classes", 1)
    checkpoint_path = model_cfg.get("checkpoint_path", None)

    # Factory mapping for easy extensibility
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

    # Optional: Load checkpoint weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Loaded pretrained model from checkpoint: {checkpoint_path}")

    return model.to(device)