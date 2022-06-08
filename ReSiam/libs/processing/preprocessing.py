import cv2
import torch
import numpy as np
import torch.nn.functional as F

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def sample_patch_multiscale(im, pos, scales, base_sample_sz, output_sz=None, constant_value=0):
    if isinstance(scales, (int, float)):
        scales = (scales,)

    # Get image patches
    im_patches = torch.cat([sample_patch(im, pos, s*base_sample_sz, output_sz, constant_value) for s in scales], dim=0)

    return  im_patches


def sample_patch(im: np.array, pos: torch.tensor, sample_sz: torch.tensor, output_sz: torch.tensor = None, constant_value=0):
        """Sample an image patch.
        args:
            im: Image
            pos: center position of crop
            sample_sz: size to crop
            output_sz: size to resize to
            max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
        """
        
        pos = torch.floor(pos)
        sz = torch.max(sample_sz.round(), torch.tensor([2]))
        
        # Extract region
        xs_min = int(pos[1] - torch.floor((sz[1] + 1) / 2))
        xs_max = int(pos[1] + sz[1] - torch.floor((sz[1] + 1) / 2))
        ys_min = int(pos[0] - torch.floor((sz[0] + 1) / 2))
        ys_max = int(pos[0] + sz[0] - torch.floor((sz[0] + 1) / 2))
        
        # Get image coordinates, (xmin, ymin, xmax, ymax)
        # patch_coord = torch.tensor([xs_min, ys_min-1, xs_max, ys_max-1]).view(1,4)
        
        # Maks sure crop region is in the image (index is in [0, im_sz) )
        xmin = max(0, xs_min)
        xmax = min(im.shape[1], xs_max)
        ymin = max(0, ys_min)
        ymax = min(im.shape[0], ys_max)
        
        im_patch = im[ymin:ymax, xmin:xmax, :]
    
        left = 0 if xmin==xs_min else int(abs(xs_min))
        right = 0 if xmax==xs_max else int(xs_max - xmax)
        top = 0 if ymin==ys_min else int(abs(ys_min))
        bottom = 0 if ymax==ys_max else int(ys_max - ymax)

        if any((left, right, top, bottom)):
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=constant_value)
        
        # Resample
        im_patch = cv2.resize(im_patch, (int(output_sz[1]), int(output_sz[0]))) \
            if output_sz is not None or (im_patch.shape[0] != output_sz[0] or im_patch.shape[1] != output_sz[1]) \
            else im_patch
        im_patch = im_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        return im_patch