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
print("Using device:", device)

def train(): 
    train_dataset = TinyKitti(root=image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn)
    
    backbone = ResNetBackBone()
    neck = Neck(backbone.outplanes)
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    
    model = CenterNet(backbone, neck, head)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
    model.train()
    
    wh_loss_factor = 1
    heat_map_loss_factor = 0.1
    wh_offset_loss_factor = 0.1
    num_epochs = 1 

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for i, sample in enumerate(train_dataloader): 
            img_list, avg_factor, target_result = sample 
            
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
            
            loss = heat_map_loss_factor*heatMap_loss + \
                wh_loss_factor*wh_loss + wh_offset_loss_factor*wh_offset_loss
            total_loss += loss.item()
            
            loss.backward
            optimizer.step()     

            if (i+1) % 1 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                %(epoch+1, num_epochs, i+1, len(train_dataloader), loss.data, total_loss / (i+1)))
                
        torch.save(model.state_dict(),'./outputs/' + str(epoch+1) + '_epoch.pth')

if __name__ == "__main__": 
    train()