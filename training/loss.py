import torch
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, targets_one_hot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # probs: (B, C, H, W)  targets_one_hot: (B, C, H, W)
    intersection = (probs * targets_one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = 0.5,
    num_classes: int = 6,
) -> torch.Tensor:
    ce = F.cross_entropy(pred, target, weight=class_weights)
    probs = pred.softmax(dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dice = dice_loss(probs, target_one_hot)
    return alpha * ce + (1 - alpha) * dice
