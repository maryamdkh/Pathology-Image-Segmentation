import segmentation_models_pytorch as smp
import torch

class LossFactory:
    def __init__(self, config):
        self.config = config["loss"]
        self.loss_fn = self._build_loss()

    def _get_single_loss(self, name):
        """Return a single loss function object (not applied yet)."""
        name = name.lower()

        # --- Built-in losses from PyTorch / SMP ---
        if name == "bce":
            return torch.nn.BCEWithLogitsLoss()

        elif name == "dice":
            return smp.losses.DiceLoss(
                mode=self.config.get("mode", "binary"),
                smooth=self.config.get("smooth", 1e-5)
            )

        elif name == "focal":
            return smp.losses.FocalLoss(
                mode=self.config.get("mode", "binary"),
                gamma=self.config.get("focal_gamma", 2.0),
                alpha=self.config.get("focal_alpha", 0.8)
            )

        elif name == "jaccard":
            # Approximation of IoU / Lovász loss
            return smp.losses.JaccardLoss(
                mode=self.config.get("mode", "binary"),
                smooth=self.config.get("smooth", 1e-5)
            )

        # --- Custom implemented losses ---
        elif name == "tversky":
            alpha = self.config.get("tversky_alpha", 0.7)
            beta = self.config.get("tversky_beta", 0.3)
            smooth = self.config.get("smooth", 1e-5)
            return lambda logits, targets: self._tversky_loss(
                logits, targets, alpha=alpha, beta=beta, smooth=smooth
            )

        elif name == "focal_tversky":
            alpha = self.config.get("tversky_alpha", 0.7)
            beta = self.config.get("tversky_beta", 0.3)
            gamma = self.config.get("tversky_gamma", 0.75)
            smooth = self.config.get("smooth", 1e-5)
            return lambda logits, targets: self._focal_tversky_loss(
                logits, targets, alpha=alpha, beta=beta, gamma=gamma, smooth=smooth
            )

        else:
            raise ValueError(f"Unknown loss name: {name}")

    # ------------------------
    # Custom loss definitions
    # ------------------------
    @staticmethod
    def _tversky_loss(logits, targets, alpha=0.7, beta=0.3, smooth=1e-5):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)
        tp = torch.sum(probs * targets)
        fn = torch.sum(targets * (1 - probs))
        fp = torch.sum((1 - targets) * probs)
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        return 1 - tversky

    @staticmethod
    def _focal_tversky_loss(logits, targets, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)
        tp = torch.sum(probs * targets)
        fn = torch.sum(targets * (1 - probs))
        fp = torch.sum((1 - targets) * probs)
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        return torch.pow((1 - tversky), gamma)

    # ------------------------
    # Combined loss builder
    # ------------------------
    def _build_loss(self):
        """Return final callable loss."""
        loss_type = self.config.get("type", "single").lower()

        if loss_type == "single":
            return self._get_single_loss(self.config["primary"])

        elif loss_type == "combined":
            primary = self._get_single_loss(self.config["primary"])
            secondary = self._get_single_loss(self.config["secondary"])
            w1, w2 = self.config.get("weights", [0.5, 0.5])

            def combined_loss(logits, targets):
                l1 = primary(logits, targets)
                l2 = secondary(logits, targets)
                total = w1 * l1 + w2 * l2
                return total, {
                    self.config["primary"]: float(l1.detach().cpu().item()),
                    self.config["secondary"]: float(l2.detach().cpu().item())
                }

            return combined_loss

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    # ------------------------
    # Call method
    # ------------------------
    def __call__(self, logits, targets):
        """Return total loss and optionally components for logging."""
        if self.config["type"].lower() == "single":
            loss = self.loss_fn(logits, targets)
            return loss, {self.config["primary"]: float(loss.detach().cpu().item())}
        else:
            return self.loss_fn(logits, targets)
