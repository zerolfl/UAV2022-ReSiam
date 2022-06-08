import torch
import torch.nn as nn
import torch.nn.functional as F

from .xcorr import xcorr_fast, xcorr_depthwise, xcorr_depthwise_new


class UPChannelRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256):
        super(UPChannelRPN, self).__init__()
        
        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num
        
        self.template_cls_conv = nn.Conv2d(in_channels, in_channels * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(in_channels, in_channels * loc_output, kernel_size=3)
        
        self.search_cls_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)
        
    def forward(self, z_feat, x_feat):
        """Get the classification and bounding box regression results using template and search samples.

        Args:
            z_feat (Tensor): Features of template sample, used to construct the kernel of RPN
            x_f (Tensor): Features of search sample, used to predict the relative target status
        """
        cls_kernel = self.template_cls_conv(z_feat)
        loc_kernel = self.template_loc_conv(z_feat)

        cls_feature = self.search_cls_conv(x_feat)
        loc_feature = self.search_loc_conv(x_feat)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, z_feat, x_feat):
        """Get the depthwise cross correlation results using template and search samples.
        Illustrations in Paper SiamRPN++ Fig.3
        """
        z_feat = self.conv_kernel(z_feat)
        x_feat = self.conv_search(x_feat)
        feature = xcorr_depthwise(x_feat, z_feat)
        out = self.head(feature)
        return out

class DepthwiseRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, hidden=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, hidden, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, hidden, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

class DepthwiseXCorrNew(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorrNew, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise_new(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPNNew(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPNNew, self).__init__()
        self.cls = DepthwiseXCorrNew(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorrNew(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=[256, 256, 256], weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


# def get_model_parameters(model):
#     total_parameters = 0
#     for layer in list(model.parameters()):
#         layer_parameter = 1
#         for l in list(layer.size()):
#             layer_parameter *= l
#         total_parameters += layer_parameter
#     return total_parameters

# model = DepthwiseRPN()
# print(get_model_parameters(model))
# model = UPChannelRPN()
# print(get_model_parameters(model))
# model = MultiRPN()
# print(get_model_parameters(model))
