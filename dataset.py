from torch.utils.data import Dataset 
import cv2 
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_gaussian(xy, gauss):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xy, gauss)
    plt.show()

def gaussian_radius(det_size, min_overlap):
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

class ToyDataset(Dataset): 
    """Toy dataset for testing"""
    
    def __init__(self, img_shape=(256, 256), max_radius=64, num_classes=1, max_objects=5):
        super().__init__()
        self.img_shape = np.array(img_shape)
        self.num_classes = num_classes 
        self.max_width = 64 
        self.max_height = 64 
        self.max_radius = min(img_shape)//4 
        self.max_objects = max_objects 
        
        w, h = self.img_shape//4 
        x_arr = np.arrage(w) + 0.5 
        y_arr = np.array(h) + 0.5 
        self.xy_mesh = np.stack(np.meshgrid(x_arr, y_arr))
        
    def __len__(self): 
        return 1000 
    
    def __getitem__(self, idx): 
        im = np.zeros(self.img_shape, dtype=np.float32)
        hw = (self.shape[0]//4, self.img_shape[1]//4)
        heatmap = np.zeros((self.num_classes+4, *hw), dtype=np.float32)
        
        for _ in range(np.random.randint(0, self.max_objects)): 
            x = np.random.randint(0, self.img_shape[0])
            y = np.random.randint(0, self.img_shape[1])
            radius = np.random.randint(10, self.max_radius)
            im = np.maximum(im, cv2.circle(im, (y, x), radius=radius, color=1, thickness=-1))
            
            center = np.array([x, y])/4 
            x, y = np.floor(center).astype(np.int)
            
            # use gaussian heatmap 
            sigma = gussian_radius(hw)
            dist_squared = np.sum((self.xy_mesh - center[:, None, None]) ** 2, axis=0)
            gauss = np.exp(-1 * dist_squared / (2 * sigma**2))
            heatmap[0, :, :] = np.maximum(heatmap[0, :, :], gauss)
            # or just set to 1 
            # heatmap[0, x, y] = 1 

            # size 
            heatmap[1:3, x, y] = np.array([2*radius, 2*radius])
            
            # offset 
            heatmap[3:, x, y] = center - np.floor(center)
            
        return im[None, :, :], heatmap 
            