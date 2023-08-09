import torch
from torch.utils.data import DataLoader
import os
import sys 
from torch import optim

from kitti_dataloader import TinyKitti, image_root
from model import ResNetBackBone, Neck, CenterNet, CenterNetHead
from loss import heatMapLoss, whAndOffsetLoss

def train(): 
    train_dataset = TinyKitti(root=image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    
    backbone = ResNetBackBone()
    
    num_deconv_filters = [256, 128, 64]
    num_deconv_kernels = [4]*3 
    neck = Neck(backbone.outplanes, num_deconv_filters, num_deconv_filters)
    
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    
    model = CenterNet(backbone, neck, head)
    
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    
    wh_loss_factor = 1
    heat_map_loss_factor = 1
    wh_offset_loss_factor = 1

    for img_list, avg_factor, target_result in train_dataloader: 
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_weight = target_result['offset_target']
        
        optimizer.zero_grad()
        
        center_heatmap_pred, wh_pred, offset_pred = model(img_list)
        
        heatMap_loss = heatMapLoss(
            center_heatmap_pred, center_heatmap_target, avg_factor)
        wh_loss = whAndOffsetLoss(
            wh_pred, wh_target, wh_offset_weight, avg_factor)
        wh_offset_loss = whAndOffsetLoss(
            offset_pred, offset_target, wh_offset_weight, avg_factor)
        
        total_loss = heat_map_loss_factor*heatMap_loss + \
            wh_loss_factor*wh_loss + wh_offset_loss_factor*wh_offset_loss
        print(f"{total_loss=}")
        
        total_loss.backward
        
        optimizer.step()        
    
if __name__ == "__main__": 
    train()