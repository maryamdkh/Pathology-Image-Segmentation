import torch

# Dice score and Dice loss for binary segmentation
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    pred: logits or probabilities, shape [B,1,H,W] or [B,H,W]
    target: binary {0,1}, shape [B,1,H,W] or [B,H,W]
    returns mean dice over batch
    """
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()