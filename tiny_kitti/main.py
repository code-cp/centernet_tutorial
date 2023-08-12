import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys 
from torch import optim

from kitti_dataloader import TinyKitti, image_root
from loss import heatMapLoss, whAndOffsetLoss
from model import ResNetBackBone, Neck, CenterNet, CenterNetHead

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(): 
    train_dataset = TinyKitti(root=image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn)
    
    backbone = ResNetBackBone()
    neck = Neck(backbone.outplanes)
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    
    model = CenterNet(backbone, neck, head)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    model.train()
    
    wh_loss_factor = 1
    heat_map_loss_factor = 1
    wh_offset_loss_factor = 1

    for img_list, avg_factor, target_result in train_dataloader: 
        img_list.to(device)
        
        center_heatmap_target = target_result['center_heatmap_target'].to(device)
        wh_target = target_result['wh_target'].to(device)
        offset_target = target_result['offset_target'].to(device)
        wh_offset_weight = target_result['offset_target'].to(device)
        
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