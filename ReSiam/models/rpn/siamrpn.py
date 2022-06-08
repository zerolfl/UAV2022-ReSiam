import torch
import torch.nn as nn

from .neck import AdjustLayerAlex, AdjustLayerConv
from .rpn_net import DepthwiseRPN, UPChannelRPN, DepthwiseRPNNew
from models.backbone import AlexNet, AlexNetLegacy, AlexNetNew


class SiamRPN(nn.Module):
    def __init__(self, backbone, neck, rpn_head):
        super(SiamRPN, self).__init__()
        self.backbone = backbone
        self.neck = neck  # For cropping backbone features
        self.rpn_head = rpn_head
        self.cell_size = 8  # stride
        
    def forward(self, exemplar_img, instance_img):
        exemplar_img = exemplar_img[:, [2,1,0], :, :]  # rgb->bgr format
        instance_img = instance_img[:, [2,1,0], :, :]  # rgb->bgr format
        # Extract backbone features
        z_feat = self.backbone(exemplar_img) if self.neck is None else self.neck(self.backbone(exemplar_img))
        x_feat = self.backbone(instance_img)
        cls, loc = self.rpn_head(z_feat, x_feat)
        return cls, loc

    def get_exemplar_feat(self, exemplar_img, use_gpu=True):
        exemplar_img = exemplar_img[:, [2,1,0], :, :]  # rgb->bgr format
        with torch.no_grad():
            self.z_feat = self.backbone(exemplar_img.cuda()) if use_gpu else self.backbone(exemplar_img)

        if self.neck is not None:
            self.z_feat = self.neck(self.z_feat)
        
    def extract(self, instance_img, use_gpu=True):
        instance_img = instance_img[:, [2,1,0], :, :]  # rgb->bgr format
        with torch.no_grad():
            self.x_feat = self.backbone(instance_img.cuda()) if use_gpu else self.backbone(instance_img)
        return self.x_feat
    
    def track(self):
        cls, loc = self.rpn_head(self.z_feat, self.x_feat)
        return cls, loc

    def size(self, im_sz: torch.tensor):
        """Feature size of backbone features"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.MaxPool2d):
                padding = torch.tensor(module.padding)
                kernel_size = torch.tensor(module.kernel_size)
                stride = torch.tensor(module.stride)
                im_sz = torch.floor((im_sz + 2 * padding - kernel_size) / stride) + 1
        return im_sz
        

def siamrpn_alexnet(use_alexnet_pad=False):
    if use_alexnet_pad:
        backbone = AlexNetNew()
        neck = AdjustLayerConv()
        rpn_head = DepthwiseRPNNew()
    else:
        backbone = AlexNet()
        neck = None
        rpn_head = DepthwiseRPN()
    model = SiamRPN(backbone, neck, rpn_head)
    return model
