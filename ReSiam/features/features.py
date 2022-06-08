import os
import torch
import torch.nn.functional as F

from torchvision.transforms.functional import rgb_to_grayscale
from .fhog import _gradient


class GrayscaleFeature:
    def __init__(self, cell_size=4):
        self.cell_size = cell_size

    def extract(self, im: torch.tensor, use_gpu=True):
        gray_feat = rgb_to_grayscale(im)
        
        if self.cell_size != 1:
            gray_feat = F.avg_pool2d(gray_feat, self.cell_size, self.cell_size)
        
        return gray_feat

    def size(self, im_sz):
        im_sz = torch.div(im_sz, self.cell_size, rounding_mode='floor')
        return im_sz

class ColornamesFeature:
    def __init__(self, cell_size=4):
        self.cell_size = cell_size
        self._table_name = 'CNnorm'
        self._factor = 32
        self._den = 8
        _dir_path = os.path.dirname(os.path.realpath(__file__))
        self._table = torch.load(os.path.join(_dir_path, 
                                              "cn_lookup_tables", 
                                              self._table_name+".pt"))

    def extract(self, im: torch.tensor, use_gpu=True):
        H, W = im.shape[-2], im.shape[-1]
        index_list = [torch.div(im[i][0, :, :].int(), self._den, rounding_mode='floor') + 
                      torch.div(im[i][1, :, :].int(), self._den, rounding_mode='floor') * self._factor + 
                      torch.div(im[i][2, :, :].int(), self._den, rounding_mode='floor') * self._factor * self._factor 
                      for i in range(im.shape[0])]
        cn_feat = [(self._table[index.flatten().long()].view(H, W, self._table.shape[1])).permute(2,0,1)
                    for index in index_list]
        cn_feat = torch.stack(cn_feat, dim=0)
        
        if self.cell_size != 1:
            cn_feat = F.avg_pool2d(cn_feat, self.cell_size, self.cell_size)
        
        return cn_feat
    
    def size(self, im_sz):
        im_sz = torch.div(im_sz, self.cell_size, rounding_mode='floor')
        return im_sz


class FhogFeature:
    def __init__(self, cell_size=4, num_orients=9, soft_bin=-1, clip=0.2):
        self.cell_size = cell_size
        self.num_orients = num_orients
        self.soft_bin = soft_bin
        self.clip = clip
    
    def extract(self, im: torch.tensor, use_gpu=True):
        M_O_list = [_gradient.gradMag(im[i].permute(1,2,0), 0, True) 
                    for i in range(im.shape[0])]
        fhog_feat_list = [_gradient.fhog(M_O[0], M_O[1],
                                         self.cell_size, 
                                         self.num_orients, 
                                         self.soft_bin, 
                                         self.clip) 
                          for M_O in M_O_list]
        fhog_feat_list = [torch.from_numpy(fhog_feat[:, :, :-1]).permute(2,0,1) 
                          for fhog_feat in fhog_feat_list]
        fhog_feat = torch.stack(fhog_feat_list, dim=0)
        
        return fhog_feat

    def size(self, im_sz):
        im_sz = torch.div(im_sz, self.cell_size, rounding_mode='floor')
        return im_sz
