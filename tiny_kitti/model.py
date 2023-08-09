import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBackBone(nn.Module): 
    r"""
    we will use a resnet 18 by default
    we will remove the last two blocks, the adaptive pool and fc layer
    """
    def __init__(self, pretrained=True): 
        super().__init__()
        resnet = resnet18(weights = ResNet18_Weights.DEFAULT).eval()
        block_list = list(resnet.children())
        self.featuremap = nn.Sequential(*block_list[:-2])
        self.outplanes = 512 
        
    def forward(self, x): 
        out = self.featuremap(x)
        return out 
    
class Neck(nn.Module): 
    def __init__(self, in_channels, num_deconv_filters, num_deconv_kernels): 
        super().__init__()
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters, num_deconv_kernels)
        
    def ConvModule(self, in_channels, feat_channels, kernel_size, stride, padding): 
        conv_layers = [
            nn.Conv2d(in_channels, feat_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=False), 
            nn.BatchNorm2d(feat_channels), 
            nn.ReLU(inplace=True)
        ]
        return conv_layers 
    
    def DeconvModule(self, in_channels, feat_channels, kernel_size, stride, padding): 
        deconv_layers = [
            nn.ConvTranspose2d(
                in_channels, feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ), 
            nn.BatchNorm2d(feat_channels), 
            nn.ReLU(inplace=True)
        ]
        return deconv_layers 
    
    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels): 
        r"""
        use deconv layers to upsample backbone's output
        """
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channels = num_deconv_filters[i]
            conv_module = self.ConvModule(
                self.in_channels, 
                feat_channels,
                3, 
                stride=1, 
                padding=1 
            )
            layers.extend(conv_module)
            upsample_module = self.DeconvModule(
                feat_channels, 
                feat_channels, 
                num_deconv_kernels[i], 
                stride=2, 
                padding=1, 
            )
            layers.extend(upsample_module)
            self.in_channels = feat_channels 
            
        return nn.Sequential(*layers)
    
    def forward(self, x): 
        out = self.deconv_layers(x)
        return out 
    
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
    test_input = torch.randn((2, 3, 512, 512))
    backbone = ResNetBackBone()
    backbond_out = backbone(test_input)
    print(f"backbone output {backbond_out.shape}")
    
    # deconv layer 
    num_deconv_filters = [256, 128, 64]
    num_deconv_kernels = [4]*3 
    neck = Neck(backbone.outplanes, num_deconv_filters, num_deconv_kernels)
    neck_out = neck(backbond_out)
    
    # head output 
    head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=3)
    heatmap, wh, offset = head(neck_out)
    
    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("Offset shape", offset.shape)

    print("==================================================")

    ct = CenterNet(backbone, neck, head)
    heatmap, wh, offset = ct(test_input)
    
    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("Offset shape", offset.shape)
        
