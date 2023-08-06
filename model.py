import torch 
import torch.nn as nn 

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1): 
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation
                     , groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_channels, out_channels, stride=1): 
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module): 
    expansion = 1 
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None): 
        super(BasicBlock, self).__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        
        if groups != 1: 
            raise ValueError('BasicBlock only supports groups=1')
        if dilation > 1: 
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample 
        self.stride = stride 
        
    def forward(self, x): 
        identity = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None: 
            identity = self.downsample(x) 
            
        out += identity 
        out = self.relu(out)
        
        return out 

# ref https://zhuanlan.zhihu.com/p/98692254
class Bottleneck(nn.Module): 
    expansion = 4 
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None): 
        super(Bottleneck, self).__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d 
            
        width = int(out_channels * (base_width / 64.)) * groups 
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample 
        self.stride = stride 
        
    def forward(self, x): 
        identity = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None: 
            identity = self.downsample(x) 
            
        out += identity 
        out = self.relu(out)
        
        return out     
        