# encoders/register_timm.py
from segmentation_models_pytorch.encoders import encoders
from models.encoders.timm_encoder import TIMMEncoder


def register_timm_encoder(alias: str, model_name: str):
    """
    Register a TIMM model under a custom SMP encoder name.
    """
    encoders[alias] = {
        "encoder": TIMMEncoder,
        "pretrained_settings": {"imagenet": {}},
        "params": {
            "model_name": model_name,
            "pretrained": True
        }
    }


def register_common_transformers():
    """
    Register a set of popular TIMM transformer backbones.
    Add or remove entries as needed.
    """
    register_timm_encoder("timm_swin_tiny",  "swin_tiny_patch4_window7_224")
    register_timm_encoder("timm_swin_small", "swin_small_patch4_window7_224")
    register_timm_encoder("timm_swin_base",  "swin_base_patch4_window7_224")

    register_timm_encoder("timm_vit_base",   "vit_base_patch16_224")
    register_timm_encoder("timm_vit_large",  "vit_large_patch16_224")

    register_timm_encoder("timm_maxvit_tiny", "maxvit_tiny_rw_224")

    register_timm_encoder("timm_convnextv2",  "convnextv2_tiny.fb_in1k")

    # Add any TIMM model name here
