import torch 
import torch.nn.functional as F 

from utils import _sigmoid

def regr_loss(regr, gt_regr, mask): 
    num = mask.float().sum()*2 
    
    regr = regr[mask==1]
    gt_regr = gt_regr[mask==1]
    regr_loss = F.l1_loss(regr, gt_regr, reduction="sum")
    
    if num != 0:
        regr_loss /= num 
    else: 
        print("mask sum is zero")
    
    return regr_loss

def _neg_loss(pred, gt, alpha=2, beta=4): 
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # NOTE, this should be very very small 
    eplison = 1e-30
    
    pos_ind = gt.eq(1).float()
    neg_ind = gt.lt(1).float()
        
    loss = 0 
    
    pos_loss = -1 * torch.log(pred+eplison) * torch.pow(1-pred, alpha) * pos_ind 
    
    neg_weights = torch.pow(1-gt, beta)
    neg_loss = -1 * torch.log(1-pred+eplison) * torch.pow(pred, alpha) * neg_weights * neg_ind 
    
    num_pos = pos_ind.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    # NOTE, must take care of num_pos == 0 case 
    # eg when gt max is 0.995, num_pos is 0 
    if num_pos == 0:
        loss = loss + neg_loss
    else:
        loss = loss + (pos_loss + neg_loss) / num_pos
        
    return loss  

def criterion(prediction, gt): 
    # need to use _sigmoid
    # ref https://github.com/xingyizhou/CenterNet/issues/234
    # pred_mask = _sigmoid(prediction[:, 0])
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = _neg_loss(pred_mask[:, None, :, :], gt[:, 0:1, :, :])
    
    size_loss_x = regr_loss(prediction[:, 1, :, :], gt[:, 1, :, :], gt[:, 0])
    size_loss_y = regr_loss(prediction[:, 2, :, :], gt[:, 2, :, :], gt[:, 0])
    size_loss = size_loss_x + size_loss_y
    
    offset_loss_x = regr_loss(prediction[:, 3, :, :], gt[:, 3, :, :], gt[:, 0])
    offset_loss_y = regr_loss(prediction[:, 4, :, :], gt[:, 4, :, :], gt[:, 0])
    offset_loss = offset_loss_x + offset_loss_y 
    
    return mask_loss, size_loss, offset_loss  