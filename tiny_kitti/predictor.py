import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys 
from torch import optim
import matplotlib
import matplotlib.pyplot as plt

import numpy as np 
import cv2 

from kitti_dataloader import TinyKitti, image_root
from model import ResNetBackBone, Neck, CenterNet, CenterNetHead

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

class Predictor: 
    def __init__(self, use_gpu):
        # input image size
        self.inp_width_  = 512
        self.inp_height_ = 512

        # confidence threshold
        self.thresh_ = 0.1

        self.use_gpu_ = use_gpu
        
    def nms(self, heat, kernel=3):
        r"""
        Non-maximal supression
        """
        pad = (kernel - 1) // 2
        hmax = torch.nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        # hmax == heat when this point is local maximal
        keep = (hmax == heat).float()
        return heat * keep
    
    def find_top_k(self, heat, K):
        ''' Find top K key points (centers) in the headmap
        '''
        batch, cat, height, width = heat.size()
        topk_scores, topk_inds = torch.topk(heat.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds // width).int().float()
        topk_xs   = (topk_inds % width).int().float() 
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_inds = gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_ys, topk_xs

    def post_process(self, xs, ys, wh, reg):
        ''' (Will modify args) Transfer all xs, ys, wh from heatmap size to input size
        '''
        for i in range(xs.size()[1]):
            xs[0, i, 0] = xs[0, i, 0] * 4
            ys[0, i, 0] = ys[0, i, 0] * 4
            wh[0, i, 0] = wh[0, i, 0] * 4
            wh[0, i, 1] = wh[0, i, 1] * 4
            
    def ctdet_decode(self, heads, K = 40):
        ''' Decoding the output

            Args:
                heads ([heatmap, width/height, regression]) - network results
            Return:
                detections([batch_size, K, [xmin, ymin, xmax, ymax, score]]) 
        '''
        heat, wh, reg = heads

        batch, cat, height, width = heat.size()

        if (not self.use_gpu_):
            plot_heapmap(heat[0,0,:,:])

        heat = self.nms(heat)

        if (not self.use_gpu_):
            plot_heapmap(heat[0,0,:,:])

        scores, inds, ys, xs = self.find_top_k(heat, K)
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        self.post_process(xs, ys, wh, reg)
        
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, 
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores], dim=2)
        
        return detections
    
    def draw_bbox(self, image, detections):
        ''' Given the original image and detections results (after threshold)
            Draw bounding boxes on the image
        '''
        height = image.shape[0]
        width = image.shape[1]
        inp_image = cv2.resize(image,(self.inp_width_, self.inp_height_))
        for i in range(detections.shape[0]):
            cv2.rectangle(inp_image, \
                        (detections[i,0],detections[i,1]), \
                        (detections[i,2],detections[i,3]), \
                        (0,255,0), 1)

        original_image = cv2.resize(inp_image,(width, height))

        return original_image
    
    def process(self, target_result, images):
        ''' The prediction process

            Args:
                images - input images (preprocessed)
            Returns:
                output - result from the network
        '''
        with torch.no_grad():
            heatmap, wh, offset = model(images)
            heatmap = heatmap.sigmoid_()

            # Generate GT data for testing
            # heatmap, wh, offset = generate_gt_data(target_result)
            
            heads = [heatmap, wh, offset]
            if (self.use_gpu_):
                torch.cuda.synchronize()
            dets = self.ctdet_decode(heads, 40) # K is the number of remaining instances

        return heads, dets

    def input2image(self, detection):
        ''' Transform the detections results from input coordinate (512*512) to original image coordinate

            x is in width direction, y is height
        '''
        default_resolution = [375, 1242]
        det_original = np.copy(detection)
        det_original[:, 0] = det_original[:, 0] / self.inp_width_ * default_resolution[1]
        det_original[:, 2] = det_original[:, 2] / self.inp_width_ * default_resolution[1]
        det_original[:, 1] = det_original[:, 1] / self.inp_width_ * default_resolution[0]
        det_original[:, 3] = det_original[:, 3] / self.inp_width_ * default_resolution[0]

        return det_original

def plot_heapmap(heatmap):
    ''' Plot the predicted heatmap

        Args:
            heatmap ([h, w]) - the heatmap output from keypoint estimator
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_title("Prediction Heatmap")
    fig.tight_layout()
    plt.show()

def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def generate_gt_data(target_result):
    ''' Generate GT data as a detection result for testing
    '''
    center_heatmap_target = target_result['center_heatmap_target'].to(device)
    wh_target = target_result['wh_target'].to(device)
    offset_target = target_result['offset_target'].to(device)

    return center_heatmap_target, wh_target, offset_target

if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    print("Use CUDA? ", use_gpu)
    
    backbone = ResNetBackBone()
    neck = Neck(backbone.outplanes)
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    
    model = CenterNet(backbone, neck, head)
    epoch = 0 
    model.load_state_dict(torch.load(f'./outputs/{str(epoch+1)}_epoch.pth'))
    model.to(device)
    
    model.eval()
    
    test_dataset = TinyKitti(root=image_root, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)

    my_predictor = Predictor(use_gpu)

    for i, sample in enumerate(test_dataloader):
        img_list, avg_factor, target_result = sample 
        img_list.to(device)
        
        # predict the output
        output, dets = my_predictor.process(target_result, img_list)                
        
        # transfer to numpy, and reshape [batch_size, K, 5] -> [K, 5]
        # only considered batch size 1 here
        dets_np = dets.detach().cpu().numpy()[0]
        
        # select detections above threshold
        threshold_mask = (dets_np[:, -1] > my_predictor.thresh_)
        dets_np = dets_np[threshold_mask, :]
        dets_original = my_predictor.input2image(dets_np)
        # if use gt 
        # dets_original = dets_np 
        
        # draw the result
        image_path = test_dataset.data_infos[i]["filename"]
        original_image = cv2.imread(image_path)
        for i in range(dets_original.shape[0]):
            cv2.rectangle(original_image, \
                        (int(dets_original[i,2]),int(dets_original[i,3])), \
                        (int(dets_original[i,0]),int(dets_original[i,1])), \
                        (0,255,0), 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.show()
        
        