import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch import optim

class ResNetBackBone(nn.Module): 
    r"""
    we will use a resnet 18 by default
    we will remove the last two blocks, the adaptive pool and fc layer
    """
    def __init__(self, pretrained=True): 
        super().__init__()
        resnet = resnet50(weights = ResNet50_Weights.DEFAULT).eval()
        block_list = list(resnet.children())
        self.featuremap = nn.Sequential(*block_list[:-2])
        self.outplanes = 2048 
        
    def forward(self, x): 
        out = self.featuremap(x)
        return out 
    
class Neck(nn.Module): 
    def __init__(self, in_channels, num_deconv_filters, num_deconv_kernels): 
        super().__init__()
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.num_layers = len(num_deconv_filters)
        self.in_channels = in_channels
        self.bn_momentum = 0.1
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters, num_deconv_kernels)
            
    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels): 
        r"""
        use deconv layers to upsample backbone's output
        """
        layers = []
        for i in range(self.num_layers):
            kernel = num_deconv_kernels[i]
            num_filter = num_deconv_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filter
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.deconv_layers(x)
    
class CenterNetHead(nn.Module): 
    def __init__(
        self, 
        in_channels, 
        feat_channels, 
        num_classes, 
    ): 
        super().__init__()
        self.heatmap_head = self._build_head(
            in_channels, 
            feat_channels, 
            num_classes, 
        )
        self.wh_head = self._build_head(
            in_channels, 
            feat_channels, 
            2
        )
        self.offset_head = self._build_head(
            in_channels, 
            feat_channels, 
            2
        )
        
    def _build_head(self, in_channels, feat_channels, out_channels): 
        r"""
        Build head for each branch
        """
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(feat_channels, out_channels, kernel_size=1)
        )
        return layer 
    
    def forward(self, feat): 
        r"""Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        cneter_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return cneter_heatmap_pred, wh_pred, offset_pred
    
class CenterNet(nn.Module): 
    def __init__(self, backbone, neck, head): 
        super().__init__()
        self.backbone = backbone 
        self.neck = neck 
        self.head = head 
        
    def forward(self, x): 
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)
        return out 
    
if __name__ == "__main__": 
    test_input = torch.randn((1, 3, 512, 512))
    backbone = ResNetBackBone()
    backbond_out = backbone(test_input)
    print(f"backbone output {backbond_out.shape}")
    
    # deconv layer 
    num_deconv_filters = [256, 128, 64]
    num_deconv_kernels = [4]*3 
    neck = Neck(backbone.outplanes, num_deconv_filters, num_deconv_kernels)
    neck_out = neck(backbond_out)
    print(f"neck_out output {neck_out.shape}")
    
    # head output 
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    heatmap, wh, offset = head(neck_out)
    
    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("Offset shape", offset.shape)

    print("==================================================")

    model = CenterNet(backbone, neck, head)
    heatmap, wh, offset = model(test_input)
    
    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("Offset shape", offset.shape)
        
