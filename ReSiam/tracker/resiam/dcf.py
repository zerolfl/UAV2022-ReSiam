import math
import torch
import numpy as np
import torch.nn.functional as F

from utils import TensorList, tensor_operation
from libs import sample_patch_multiscale
from features import FhogFeature

@tensor_operation
def fft(a: torch.Tensor, dim = (-1)):
    return torch.fft.fft(a, dim = dim)


@tensor_operation
def ifft(a: torch.Tensor, dim = (-1)):
    return torch.fft.ifft(a, dim = dim)


@tensor_operation
def fft2(a: torch.Tensor, dim = (-2, -1)):
    return torch.fft.fft2(a, dim = dim)


@tensor_operation
def ifft2(a: torch.Tensor, dim = (-2, -1)):
    return torch.fft.ifft2(a, dim = dim)


@tensor_operation
def qr(a: torch.Tensor):
    """Do (reduced) QR decomposition
    """
    return torch.linalg.qr(a, mode='reduced')[0]
    # return torch.from_numpy(numpy.linalg.qr(a, mode='reduced')[0])
    # return torch.from_numpy(linalg.qr(a, mode='economic')[0])


@tensor_operation
def resize_dft(inputdft, desired_len):
    """Resize a one-dimensional DFT to the desired length.
    """
    input_len = len(inputdft)
    minsz = min(input_len, desired_len)

    scaling = desired_len / input_len

    resize_dft = torch.zeros(desired_len, dtype=inputdft.dtype)

    mids = int(math.ceil((minsz + 1) / 2))
    mide = int(math.floor((minsz - 1) / 2))

    resize_dft[:mids] = scaling * inputdft[:mids]
    resize_dft[-mide:] = scaling * inputdft[-mide:]
    
    return resize_dft


@tensor_operation
def resize_dft2(inputdft, desired_sz):
    """Resize a two-dimensional DFT to the desired size.
    """
    input_sz = torch.Tensor(list(inputdft.shape[-2:]))  # H, W
    
    if any(input_sz != desired_sz):
        minsz = torch.min(input_sz, desired_sz)

        scaling = torch.prod(desired_sz) / torch.prod(input_sz)
        
        desired_shape = list(inputdft.shape[:-2])
        desired_shape.extend(desired_sz.long().tolist())
        
        resize_dft = torch.zeros(desired_shape, 
                                 dtype=inputdft.dtype, 
                                 device=inputdft.device)
        
        mids = (torch.ceil((minsz + 1) / 2)).long().tolist()
        mide = (torch.floor((minsz - 1) / 2)).long().tolist()
        
        resize_dft[..., :mids[0], :mids[1]] = scaling * inputdft[..., :mids[0], :mids[1]]
        resize_dft[..., :mids[0], -mide[1]:] = scaling * inputdft[..., :mids[0], -mide[1]:]
        resize_dft[..., -mide[0]:, :mids[1]] = scaling * inputdft[..., -mide[0]:, :mids[1]]
        resize_dft[..., -mide[0]:, -mide[1]:] = scaling * inputdft[..., -mide[0]:, -mide[1]:]
    else:
        resize_dft = inputdft
        
    return resize_dft


def norm_response(response, norm_mode='relative_sum'):
        """Do softmax along with dim=(-2,-1). Only supports when reponse's dim=[B, H, W]
        """
        if norm_mode == 'relative_max':
            max_value = torch.max(torch.max(response, dim=-2)[0], dim=-1)[0].view(-1, 1, 1)
            response = response / max_value
        elif norm_mode == 'relative_sum':
            sum_value = torch.sum(response, dim=(-2,-1)).view(-1, 1, 1)
            response = response / sum_value
        elif norm_mode == 'softmax':
            max_value = torch.max(torch.max(response, dim=-2)[0], dim=-1)[0].view(-1, 1, 1)
            exp_num = torch.exp(response - max_value)
            exp_dom = torch.sum(exp_num, dim=(-2,-1)).view(-1, 1, 1)
            response = exp_num / exp_dom
            response = response / torch.max(torch.max(response, dim=-2)[0], dim=-1)[0].view(-1, 1, 1)
        elif norm_mode == 'softmax_element':
            max_value = torch.max(torch.max(response, dim=-2)[0], dim=-1)[0].view(-1, 1, 1)
            response = response / max_value
            background_resp = 1 - response
            cls_flatten = torch.cat([response.view(1, -1), background_resp.view(1, -1)], dim=0)
            cls_flatten = F.softmax(cls_flatten, dim=0).data[0, :]
            response = cls_flatten.view(-1, response.shape[-2], response.shape[-1])
        return response


@tensor_operation
def feat_grid(sz):
    rg = torch.roll(
        torch.arange(-torch.floor((sz[0]-1)/2).int(), torch.ceil((sz[0]-1)/2) + 1).int(),
        -torch.floor(torch.floor((sz[0]-1)/2)).int().item())
    cg = torch.roll(
        torch.arange(-torch.floor((sz[1]-1)/2).int(), torch.ceil((sz[1]-1)/2) + 1).int(),
        -torch.floor(torch.floor((sz[1]-1)/2)).int().item())
    
    return rg, cg


@tensor_operation
def label_function_spatial(sz, sigma, negative=False):
    """Returns the gaussian label y, whose dim is [None, None, H, W]
    """
    rg, cg = feat_grid(sz)
    try:
        rs, cs = torch.meshgrid(rg, cg, indexing='ij')
    except:
        rs, cs = torch.meshgrid(rg, cg)
    y = torch.exp(-0.5 * ((rs ** 2 + cs ** 2) / sigma ** 2))
    y = torch.exp(-0.5 * ((rs ** 2 + cs ** 2) / sigma ** 2))
    if negative:
        y = torch.zeros_like(y)
    return y[None, None, :, :]


@tensor_operation
def cosine_window(sz):
    """Returns the cos window, whose dim is [None, None, H, W]
    """
    cos_window = torch.hann_window(int(sz[0]+2), periodic=False)[:, None].mm(
        torch.hann_window(int(sz[1]+2), periodic=False)[None, :])
    cos_window = cos_window[None, None, 1:-1, 1:-1]
    return cos_window


@tensor_operation
def reg_window(sz, target_sz, reg_window_max, reg_window_min, negative=False):
    reg_scale = torch.floor(target_sz).int()
    reg_window = torch.ones([sz[0].int(), sz[1].int()]) * reg_window_max
    if not negative:
        _range = torch.zeros([2, 2])
        for j in range(2):
            _range[j, :] = torch.Tensor([0, reg_scale[j] - 1]) - torch.floor(reg_scale[j] / 2)
        center = torch.floor((sz+ 1) / 2) + (sz + 1) % 2 - 1
        range_hmin, range_hmax = int(center[0] + _range[0, 0]), int(center[0] + _range[0, 1] + 1)
        range_wmin, range_wmax = int(center[1] + _range[1, 0]), int(center[1] + _range[1, 1] + 1)
        reg_window[range_hmin:range_hmax, range_wmin:range_wmax] = reg_window_min
        
    return reg_window


@tensor_operation
def shift_sample(xf, ky, kx, translation_shift):
    """Shift sample in Fourier domain
    """
    signal_size = torch.Tensor([xf.shape[-2], xf.shape[-2]])
    shift = 2 * np.pi * translation_shift / signal_size
    
    shift_exp_y = torch.exp(shift[0] * ky * 1j)
    shift_exp_x = torch.exp(shift[1] * kx * 1j)
    
    shift_exp = (shift_exp_y.view(-1, 1) * shift_exp_x.view(1, -1))[None, None, :, :]
    
    xf = xf * shift_exp
    
    return xf


@tensor_operation
def train_recf(model_xf, yf, reg_window, xf_p, wf_p, params):
    Sxy = model_xf * torch.conj(yf)
    Sxx = model_xf * torch.conj(model_xf)
    Sxx_p = xf_p * torch.conj(xf_p)

    # feature size
    sz = torch.Tensor([model_xf.shape[-2], model_xf.shape[-1]])
    N = torch.prod(sz)

    # initialize hf
    hf = torch.zeros_like(model_xf)
    
    # initialize lagrangian multiplier
    zetaf = torch.zeros_like(model_xf)
    
    #  ReCF parameters
    gamma_I = params.gamma_I  # Parameter on Inferred response regularization
    gamma_H = params.gamma_H  # Parameter on historical response regularization
    
    # ADMM parameters
    mu = params.mu
    beta = params.beta
    mu_max = params.mu_max
    
    # ADMM iterations
    iter = 1
    while iter <= params.admm_iterations:
        wf = (Sxy + (gamma_I * Sxx + gamma_H * Sxx_p) * wf_p + mu * hf - zetaf) / \
             ((1 + gamma_I) * Sxx + gamma_H * Sxx_p + mu)
        hf = fft2(ifft2(mu * wf + zetaf) / (1 / N * reg_window ** 2 + mu))
        zetaf = zetaf + mu * (wf - hf)
        mu = min(mu_max, beta * mu)
        iter += 1
    return wf


@tensor_operation
def resp_newton(response, responsef, ky, kx, use_sz, iterations):
    """
    Args:
        response: dim=(B,H,W)
        responsef: dim=(B,H,W)
        ky: grid along with dim H
        kx: grid along with dim W
        use_sz: interpolation size
        iterations: iterations of Newton's method
    """
    n_scale = response.shape[0]  # num of scales
    
    max_resp_in_row, index_max_in_row = torch.max(response, dim=-2)
    init_max_response, index_max_in_col = torch.max(max_resp_in_row, dim=-1)
    
    col = index_max_in_col.flatten()
    row = index_max_in_row[torch.arange(n_scale), col]

    trans_row = (row - 1 + torch.floor((use_sz[1] - 1) / 2)) % use_sz[1] \
                - torch.floor((use_sz[1] - 1) / 2) + 1
    trans_col = (col - 1 + torch.floor((use_sz[0] - 1) / 2)) % use_sz[0] \
                - torch.floor((use_sz[0] - 1) / 2) + 1
                
    init_pos_y = (2 * np.pi * trans_row / use_sz[1]).view(n_scale, 1, 1)
    init_pos_x = (2 * np.pi * trans_col / use_sz[0]).view(n_scale, 1, 1)
    
    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    # pre-compute complex exponential
    iky = (1j * ky).view(1,1,-1)
    exp_iky = torch.exp(iky * max_pos_y)
    ikx = (1j * kx).view(1,-1,1)
    exp_ikx = torch.exp(ikx * max_pos_x)

    ky2 = ky * ky
    kx2 = kx * kx

    iter = 1
    while iter <= iterations:
        # Compute gradient
        ky_exp_ky = exp_iky * ky.view(1,1,-1)
        kx_exp_kx = exp_ikx * ky.view(1,-1,1)
        
        y_resp = torch.matmul(exp_iky, responsef)
        resp_x = torch.matmul(responsef, exp_ikx)
        
        grad_y = -torch.imag(torch.matmul(ky_exp_ky, resp_x))
        grad_x = -torch.imag(torch.matmul(y_resp, kx_exp_kx))
        
        H_yy = -torch.real(torch.matmul(exp_iky * ky2.view(1,1,-1), resp_x))
        H_xx = -torch.real(torch.matmul(y_resp, exp_ikx * kx2.view(1,-1,1)))
        H_xy = -torch.real(torch.matmul(ky_exp_ky, torch.matmul(responsef, kx_exp_kx)))

        det_H = H_yy * H_xx - H_xy * H_xy

        # Compute Newton's direction
        diff_y = (H_xx * grad_y - H_xy * grad_x) / det_H
        diff_x = (H_yy * grad_x - H_xy * grad_y) / det_H
        max_pos_y = max_pos_y - diff_y
        max_pos_x = max_pos_x - diff_x

        # Evaluate maximum
        exp_iky = torch.exp(iky * max_pos_y)
        exp_ikx = torch.exp(ikx * max_pos_x)

        iter = iter + 1

    max_response = torch.real(torch.matmul(exp_iky, torch.matmul(responsef, exp_ikx))) / torch.prod(use_sz)

    # Check for scales that have not increased in score
    ind = max_response.flatten() < init_max_response
    max_response[ind, 0, 0] = init_max_response[ind]
    max_pos_y[ind, 0, 0] = init_pos_y[ind, 0, 0]
    max_pos_x[ind, 0, 0] = init_pos_x[ind, 0, 0]

    sind = int(torch.argmax(max_response, 0))
    disp_row = ((max_pos_y[sind, 0, 0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[1]
    disp_col = ((max_pos_x[sind, 0, 0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[0]

    return disp_row.item(), disp_col.item(), sind, max_response[sind].item()


class ScaleFilter:
    """Scale filter used to estimate the target scale
    """
    def __init__(self, target_sz, average_channel_value, params):
        self.average_channel_value = average_channel_value
        self.scale_feature = FhogFeature()
        
        init_target_sz = target_sz
        
        num_scales = params.number_of_scales_filter
        scale_step = params.scale_step_filter
        scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor
        
        scale_exp = torch.arange(-math.floor(num_scales - 1) / 2,
                                 math.ceil(num_scales - 1) / 2 + 1) * params.number_of_interp_scales / num_scales
        scale_exp_shift = torch.roll(scale_exp, 
                                     -int(math.floor((num_scales - 1) / 2)))
        
        interp_scale_exp = torch.arange(-math.floor((params.number_of_interp_scales - 1) / 2),
                                        math.ceil((params.number_of_interp_scales - 1) / 2) + 1)
        interp_scale_exp_shift = torch.roll(interp_scale_exp, 
                                            -int(math.floor(params.number_of_interp_scales - 1) / 2))
        
        self.scale_size_factors = scale_step ** scale_exp
        self.interp_scale_factors = scale_step ** interp_scale_exp_shift
        
        # Construct the label function
        ys = torch.exp(-0.5 * (scale_exp_shift ** 2) / (scale_sigma ** 2))
        self.yf = fft(ys)[None, :]
        self.cos_window = torch.hann_window(int(ys.shape[0]), periodic=False)
        
        # make sure the scale model is not to large, to save computation time
        if torch.prod(init_target_sz) > params.scale_model_max_area:
            scale_model_factor = torch.sqrt(params.scale_model_max_area / torch.prod(init_target_sz))
        else:
            scale_model_factor = 1.0
        
        self.scale_model_sz = torch.maximum(torch.floor(init_target_sz * scale_model_factor), torch.Tensor([8, 8]))
        self.do_feat_compress = params.do_feat_compress
        
        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_factors = torch.ones(1)
    
    def update(self, image, pos, current_scale_factor, base_target_sz, params):
        """Update scale filter
        """
        # Get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        
        xs = TensorList([self.scale_feature.extract(sample_patch_multiscale(image, pos, scales, base_target_sz, self.scale_model_sz, self.average_channel_value)),])
        
        # Vectorize the feature
        xs = xs.permute(1,3,2,0).reshape(-1, self.num_scales)
        
        first_frame = not hasattr(self, 's_num')

        if first_frame:
            self.s_num = xs
        else:
            self.s_num = (1 - params.scale_learning_rate) * self.s_num + params.scale_learning_rate * xs
            
        # compute projection basis
        if self.do_feat_compress:
            self.basis = qr(self.s_num)
            scale_basis_den = qr(xs)
            self.basis = self.basis.permute(1, 0)
            scale_basis_den = scale_basis_den.permute(1, 0)
        else:
            self.basis = 1
            scale_basis_den = 1
            
        # compute numerator
        feat_proj = (self.basis @ self.s_num) * self.cos_window
        sf_proj = fft(feat_proj, dim=-1)
        self.sf_num = sf_proj.conj() * self.yf
        
        # update denominator
        xs = (scale_basis_den @ xs) * self.cos_window
        xsf = fft(xs, dim=-1)
        
        new_sf_den = TensorList([torch.sum(torch.real(_xsf * _xsf.conj()), 0) for _xsf in xsf])
        
        if first_frame:
            self.sf_den_init = new_sf_den
            self.sf_den = new_sf_den
        else:
            self.sf_den = (1 - params.scale_learning_rate) * self.sf_den + params.scale_learning_rate * new_sf_den

    def reset(self):
        self.sf_den = self.sf_den_init
    
    def track(self, image, pos, current_scale_factor, base_target_sz, params):
        # get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        xs = TensorList([self.scale_feature.extract(sample_patch_multiscale(image, pos, scales, base_target_sz, self.scale_model_sz, self.average_channel_value)),])
        
        # Vectorize the feature
        xs = xs.permute(1,3,2,0).reshape(-1, self.num_scales)
        
        # project
        xs = (self.basis @ xs) * self.cos_window
        xsf = fft(xs, dim=-1)
        
        # get scores
        scale_responsef = TensorList([torch.sum(_sf_num * _xsf, 0) / (_sf_den + params.lamBda) 
                           for _sf_num, _xsf, _sf_den in zip(self.sf_num, xsf, self.sf_den)])
        
        interp_scale_response = ifft(resize_dft(scale_responsef, params.number_of_interp_scales))
        interp_scale_response = torch.real(interp_scale_response[0])
        recovered_scale_index = torch.argmax(interp_scale_response)
        
        if params.do_poly_interp:
            # Fit a quadratic polynomial to get a refined scale estimate
            id1 = (recovered_scale_index - 1) % params.number_of_interp_scales
            id2 = (recovered_scale_index + 1) % params.number_of_interp_scales
            poly_x = torch.Tensor([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index], self.interp_scale_factors[id2]])
            poly_y = torch.Tensor([[interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]]])
            poly_A = torch.Tensor([[poly_x[0]**2, poly_x[0], 1],
                                   [poly_x[1]**2, poly_x[1], 1],
                                   [poly_x[2]**2, poly_x[2], 1]])
            poly = torch.linalg.inv(poly_A).mm(poly_y.T)
            scale_change_factor = - poly[1] / (2 * poly[0])
        else:
            scale_change_factor = self.interp_scale_factors[recovered_scale_index]
        
        return scale_change_factor
        