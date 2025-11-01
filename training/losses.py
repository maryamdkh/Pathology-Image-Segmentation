import segmentation_models_pytorch as smp
import torch

class LossFactory:
    def __init__(self, config):
        self.config = config["loss"]
        self.loss_fn = self._build_loss()

    def _get_single_loss(self, name):
        """Return a single loss function object (not applied yet)."""
        name = name.lower()
        if name == "bce":
            return torch.BCEWithLogitsLoss()
        elif name == "dice":
            return smp.losses.DiceLoss(
                mode=self.config.get("mode", "binary"),
                smooth=self.config.get("smooth", 1e-5)
            )
        elif name == "focal":
            return smp.losses.FocalLoss(
                mode=self.config.get("mode", "binary"),
                gamma=self.config.get("focal_gamma", 2.0)
            )
        else:
            raise ValueError(f"Unknown loss name: {name}")

    def _build_loss(self):
        """Return final callable loss."""
        loss_type = self.config.get("type", "single")

        if loss_type == "single":
            return self._get_single_loss(self.config["primary"])

        elif loss_type == "combined":
            primary = self._get_single_loss(self.config["primary"])
            secondary = self._get_single_loss(self.config["secondary"])
            w1, w2 = self.config.get("weights", [0.5, 0.5])

            def combined_loss(logits, targets):
                l1 = primary(logits, targets)
                l2 = secondary(logits, targets)
                return w1 * l1 + w2 * l2, {"primary": l1.item(), "secondary": l2.item()}

            return combined_loss

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def __call__(self, logits, targets):
        """Return total loss and (optionally) components for logging."""
        if self.config["type"] == "single":
            loss = self.loss_fn(logits, targets)
            return loss, {self.config["primary"]: loss.item()}
        else:
            return self.loss_fn(logits, targets)