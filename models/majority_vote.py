import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Dict, Optional


class MajorityVotingEnsemble(nn.Module):
    """
    Ensemble model that performs voting across multiple segmentation models.
    """
    
    def __init__(self, model_configs: List[Dict], device: torch.device = torch.device("cuda"),
                 voting_strategy: str = "majority", confidence_threshold: float = 0.5):
        super().__init__()
        self.device = device
        self.voting_strategy = voting_strategy
        self.confidence_threshold = confidence_threshold
        self.models = nn.ModuleList()
        
        # Build each model
        for config in model_configs:
            model = self._build_single_model(config, device)
            self.models.append(model)
        
        self.num_models = len(self.models)
        
    def _build_single_model(self, config: Dict, device: torch.device) -> nn.Module:
        """Build a single segmentation model."""
        architecture = config.get("architecture", "UnetPlusPlus")
        encoder_name = config.get("encoder_name", "resnet50")
        encoder_weights = config.get("encoder_weights", "imagenet")
        in_channels = config.get("in_channels", 3)
        num_classes = config.get("classes", 1)
        checkpoint_path = config.get("checkpoint_path", None)

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

        return model.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with selected voting strategy."""
        if self.voting_strategy == "majority":
            return self._majority_vote(x)
        elif self.voting_strategy == "average":
            return self._average_vote(x)
        elif self.voting_strategy == "max_confidence":
            return self._max_confidence_vote(x)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _majority_vote(self, x: torch.Tensor) -> torch.Tensor:
        """Majority voting implementation."""
        all_predictions = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                preds = torch.argmax(logits, dim=1) if logits.shape[1] > 1 else (torch.sigmoid(logits) > self.confidence_threshold).long()
                all_predictions.append(preds)
        
        predictions_stack = torch.stack(all_predictions, dim=0)
        majority_vote = torch.mode(predictions_stack, dim=0).values
        return majority_vote
    
    def _average_vote(self, x: torch.Tensor) -> torch.Tensor:
        """Average probability voting."""
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        
        if avg_probs.shape[1] == 1:
            return (avg_probs > self.confidence_threshold).long()
        else:
            return torch.argmax(avg_probs, dim=1)
    
    def _max_confidence_vote(self, x: torch.Tensor) -> torch.Tensor:
        """Vote based on maximum confidence."""
        # Implementation for max confidence voting
        pass

