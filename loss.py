import torch 
import torch.nn.functional as F 

epsilon = 1e-4

def regr_loss(regr, gt_regr, mask): 
    num = mask.float().sum()*2 
    
    regr = regr[mask==1]
    gt_regr = gt_regr[mask==1]
    regr_loss = F.l1_loss(regr, gt_regr, reduction="sum")
    regr_loss /= (num+epsilon)
    
    return regr_loss

def _neg_loss(pred, gt, alpha=2, beta=4): 
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_ind = gt.eq(1).float()
    neg_ind = gt.lt(1).float()
        
    loss = 0 
    
    pos_loss = -1 * torch.log(pred) * torch.pow(1-pred, alpha) * pos_ind 
    
    neg_weights = torch.pow(1-gt, beta)
    neg_loss = -1 * torch.log(1-pred) * torch.pow(pred, alpha) * neg_weights * neg_ind 
    
    num_pos = pos_ind.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    loss += (pos_loss + neg_loss) / (num_pos+epsilon) 
        
    return loss  

def criterion(prediction, gt): 
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = _neg_loss(pred_mask[:, None, :, :], gt[:, 0:1, :, :])
    
    size_loss_x = regr_loss(prediction[:, 1, :, :], gt[:, 1, :, :], gt[:, 0])
    size_loss_y = regr_loss(prediction[:, 2, :, :], gt[:, 2, :, :], gt[:, 0])
    size_loss = size_loss_x + size_loss_y
    
    offset_loss_x = regr_loss(prediction[:, 3, :, :], gt[:, 3, :, :], gt[:, 0])
    offset_loss_y = regr_loss(prediction[:, 4, :, :], gt[:, 4, :, :], gt[:, 0])
    offset_loss = offset_loss_x + offset_loss_y 
    
    return mask_loss, size_loss, offset_loss  