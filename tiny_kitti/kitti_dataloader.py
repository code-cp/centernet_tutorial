import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
from albumentations import Compose, BboxParams, RandomBrightnessContrast, GaussNoise,\
    RGBShift, CLAHE, RandomGamma, HorizontalFlip, RandomResizedCrop, Resize, Normalize
import cv2
from math import sqrt

image_root = r"./tiny_kitti/data"

class Transform(object): 
    def __init__(self, box_format='coco', height=512, width=512): 
        self.tsfm_train = Compose([
            Resize(height=height, width=width),
            HorizontalFlip(), 
            RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            GaussNoise(), 
            RGBShift(), 
            CLAHE(),
            RandomGamma(),
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels'])) 

        self.tsfm_test = Compose([
            CLAHE(),
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.0, label_fields=['labels'])) 
        
    def __call__(self, mode, image, bboxes, labels):
        if mode == "train":
            augmented = self.tsfm_train(image=image, bboxes=bboxes, labels=labels)
        else:
            augmented = self.tsfm_test(image=image, bboxes=bboxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes'] 
        return img, boxes 
    
def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cuda'): 
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)
    
    h = (-(x**2 + y**2) / (2 * sigma**2)).exp()
    
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0 
    
    return h 

def gen_gaussian_target(heatmap, center, radius, k=1): 
    r"""
    sigma is fixed 
    but radius is computed 
    """
    diameter = 2 * radius + 1 
    gaussian_kernel = gaussian2D(radius, sigma=diameter/6, dtype=heatmap.dtype, device=heatmap.device)
    
    x, y = center 
    
    height, width = heatmap.shape[:2]
    
    left, right = min(x, radius), min(width-x, radius+1) 
    top, bottom = min(y, radius), min(height-y, radius+1)
    
    # extract the rectangle centered at x, y and has width/length of radius 
    masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
    masked_gaussian = gaussian_kernel[radius-top:radius+bottom, radius-left:radius+right]
    
    out_heatmap = heatmap 
    torch.max(
        masked_heatmap,
        masked_gaussian*k, 
        out=out_heatmap[y-top:y+bottom, x-left:x+right] 
    ) 
    
    return out_heatmap 

def gaussian_radius(det_size, min_overlap=0.7):
    r"""Generate 2D gaussian radius.

    This function is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
    utils.py#L65>`_.

    Given ``min_overlap``, radius could computed by a quadratic equation
    according to Vieta's formulas.

    There are 3 cases for computing gaussian radius, details are following:

    - Explanation of figure: ``lt`` and ``br`` indicates the left-top and
      bottom-right corner of ground truth box. ``x`` indicates the
      generated corner at the limited position when ``radius=r``.

    - Case1: one corner is inside the gt box and the other is outside.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x----------+--+
        |  |          |  |
        |  |          |  |    height
        |  | overlap  |  |
        |  |          |  |
        |  |          |  |      v
        +--+---------br--+      -
           |          |  |
           +----------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
        {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
        {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case2: both two corners are inside the gt box.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x-------+  |
        |  |       |  |
        |  |overlap|  |       height
        |  |       |  |
        |  +-------x--+
        |          |  |         v
        +----------+-br         -

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
        {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
        {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case3: both two corners are outside the gt box.

    .. code:: text

           |<   width   >|

        x--+----------------+
        |  |                |
        +-lt-------------+  |   -
        |  |             |  |   ^
        |  |             |  |
        |  |   overlap   |  | height
        |  |             |  |
        |  |             |  |   v
        |  +------------br--+   -
        |                |  |
        +----------------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
        {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
        {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
        {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

    Args:
        det_size (list[int]): Shape of object.
        min_overlap (float): Min IoU with ground truth for boxes generated by
            keypoints inside the gaussian kernel.

    Returns:
        radius (int): Radius of gaussian kernel.
    """
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    # NOTE, initial version is just /2 not /2a, which is a bug
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    # NOTE, initial version is just /2 not /2a, which is a bug
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    # NOTE, initial version is just /2 not /2a, which is a bug
    r3  = (b3 + sq3) / (2 * a3)
    
    return min(r1, r2, r3)

def read_list(filename): 
    with open(filename) as file: 
        lines = file.readlines()
        lines = [line.rstrip() for line in lines] 
    return lines 

class TinyKitti(Dataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    ann_file = 'train.txt'
    img_prefix = 'training/image_2'
    
    def __init__(self, root, resize=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115),
                 std=(0.28863828, 0.27408164, 0.27809835),):
        self.down_stride = 4 
        self.num_classes = len(self.CLASSES)
        self.cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        self.image_list = read_list(os.path.join(root, self.ann_file))
        self.mode = mode 
        self.resize_size = resize 
        self.mean = mean 
        self.std = std 
        self.data_infos = []
        
        self.transform = Transform('pascal_voc', height=self.resize_size[0], width=self.resize_size[1])
        
        for image_id in self.image_list: 
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image_path = os.path.join(root, filename)
            data_info = dict(
                filename=image_path, 
            )
            
            label_prefix = self.img_prefix.replace('image_2', 'label_2') 
            lines = read_list(os.path.join(root, label_prefix, f'{image_id}.txt'))
            
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]
            
            gt_bboxes = []
            gt_labels = []
            
            for bbox_name, bbox in zip(bbox_names, bboxes): 
                if bbox_name in self.cat2label: 
                    gt_labels.append(self.cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                    
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4), 
                labels=np.array(gt_labels, dtype=np.int32)
            ) 
            
            data_info.update(ann=data_anno)
            
            self.data_infos.append(data_info)
            
    def __getitem__(self, index): 
        data_dict = self.data_infos[index]
        image_path = data_dict["filename"]
        bboxes = data_dict["ann"]["bboxes"]
        labels = data_dict["ann"]["labels"]
        image = cv2.imread(image_path)
          
        image, boxes = self.transform(self.mode, image, bboxes, labels)

        self.image_shape = image.shape 
        self.featuremap_shape = [image.shape[0]//self.down_stride,
                                image.shape[1]//self.down_stride]
       
        # change image range to 0-1 and HWC to CHW  
        self.img = transforms.ToTensor()(image)
        self.boxes = torch.Tensor(boxes)
        self.labels = torch.LongTensor(labels)
        
        return self.img, self.boxes, self.labels 
    
    def __len__(self): 
        return len(self.data_infos)
    
    def collate_fn(self, data): 
        img_list, gt_bboxes, gt_labels = zip(*data)
        
        img_h, img_w = self.image_shape[0], self.image_shape[1]
        batch_size = len(img_list)
        feat_h, feat_w = self.featuremap_shape[0], self.featuremap_shape[1]
        
        width_ratio = float(feat_w/img_w)
        height_ratio = float(feat_h/img_h)
        
        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, feat_h, feat_w]
        )
        wh_target = gt_bboxes[-1].new_zeros(
            [batch_size, 2, feat_h, feat_w]
        )
        offset_target = gt_bboxes[-1].new_zeros(
            [batch_size, 2, feat_h, feat_w]
        )
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [batch_size, 2, feat_h, feat_w]
        )
        
        for batch_id in range(batch_size): 
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            
            # this is the feature map center, not original image center
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2 
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2 
            gt_centers = torch.cat((center_x, center_y), dim=1)
            
            for j, ct in enumerate(gt_centers): 
                ctx_int, cty_int = ct.int()
                ctx, cty = ct 
                
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio 
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio 
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                
                ind = gt_label[j]
                gen_gaussian_target(
                    center_heatmap_target[batch_id, ind], 
                    [ctx_int, cty_int], 
                    radius 
                )
                
                # NOTE, y is row, x is column  
                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h 
                
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1 
                
        # the average factor is used, suppose if we have 10 bboxes in one image, then we need to average the loss of those 10 by dividing by 10
        # so the loss from an image with 2 bbox and 10 bbox will be same, otherwise the latter will contribute more.
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        avg_factor = torch.tensor(avg_factor, dtype=torch.float32)
         
        target_result = dict(
            center_heatmap_target=center_heatmap_target, 
            wh_target=wh_target, 
            offset_target=offset_target, 
            wh_offset_target_weight=wh_offset_target_weight, 
        )
        
        img_list = torch.stack(img_list)
        
        return img_list, avg_factor, target_result 
    
if __name__ == "__main__": 
    dataset = TinyKitti(root=image_root)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))
    