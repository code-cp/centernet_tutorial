import torch 
import torch.nn as nn 

def _sigmoid(x): 
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1e-4)
    return y 

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # Here, the gather function is used to extract values from the feat tensor based on the indices provided in the modified ind tensor. The 1 as the first argument of gather indicates that the indices are taken along the second dimension (axis 1) of the feat tensor
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

