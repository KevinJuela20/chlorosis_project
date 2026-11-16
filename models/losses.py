"""
MÓDULO DE FUNCIONES DE PÉRDIDA ENMASCARADAS (MASKED LOSSES)
Proyecto: Bitemporal Adapter Network (BAN) para Dinámica de Clorosis Foliar

Estrategia del Director:
- Solo se calcula el error y se optimiza la red dentro del área geométrica donde
  ambas hojas se solapan (Máscara de Solapamiento / Intersection RoI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedDiceLoss(nn.Module):
    """
    Dice Loss calculada exclusivamente sobre la región de solapamiento foliar.
    """
    def __init__(self, smooth=1e-5):
        super(MaskedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask_overlap):
        """
        Args:
            pred: Tensor (B, 1, H, W) con probabilidades [0, 1] o logits tras sigmoid.
            target: Tensor (B, 1, H, W) binario con el avance real de clorosis.
            mask_overlap: Tensor (B, 1, H, W) binario con el área de solapación del director.
        """
        # Enmascarar predicción y objetivo
        pred_m = pred * mask_overlap
        target_m = target * mask_overlap

        pred_flat = pred_m.view(pred.size(0), -1)
        target_flat = target_m.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class MaskedFocalLoss(nn.Module):
    """
    Focal Loss ponderada por la máscara de solapamiento para resolver el desbalance
    entre tejido sano y pequeñas lesiones incipientes de clorosis.
    """
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target, mask_overlap):
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # Cross Entropy por pixel
        pt = torch.where(target == 1, pred, 1.0 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)

        focal_loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)

        # Aplicar máscara de solapamiento
        masked_focal = focal_loss * mask_overlap
        denom = mask_overlap.sum().clamp(min=1.0)
        return masked_focal.sum() / denom


class MaskedCompoundLoss(nn.Module):
    """
    Pérdida Compuesta de alta fidelidad para Supercomputador:
    L = weight_focal * MaskedFocalLoss + weight_dice * MaskedDiceLoss
    """
    def __init__(self, weight_focal=1.0, weight_dice=1.0):
        super(MaskedCompoundLoss, self).__init__()
        self.focal = MaskedFocalLoss()
        self.dice = MaskedDiceLoss()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, pred, target, mask_overlap):
        loss_focal = self.focal(pred, target, mask_overlap)
        loss_dice = self.dice(pred, target, mask_overlap)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice, loss_focal, loss_dice

