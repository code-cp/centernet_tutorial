import torch.nn as nn
import torch

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0): 
    r"""`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    
    # NOTE, need to use eps in log 
    eps = torch.finfo(pred.dtype).eps
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights 
    
    return pos_loss + neg_loss 

def heatMapLoss(pred, gaussian_target, avg_factor): 
    loss = gaussian_focal_loss(pred, gaussian_target)
    eps = torch.finfo(avg_factor.dtype).eps 
    loss = loss.sum() / (avg_factor + eps)
    return loss 

def l1_loss(pred, target): 
    r"""L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0: 
        return pred.sum() * 0 
    
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    
    return loss 

def whAndOffsetLoss(pred, target, weight, avg_factor): 
    loss = l1_loss(pred, target) * weight 
    
    eps = torch.finfo(avg_factor.dtype).eps 
    # Since the channel of wh_target and offset_target is 2, the avg_factor
    # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
    loss = loss.sum() / (avg_factor*2 + eps)
    
    return loss 