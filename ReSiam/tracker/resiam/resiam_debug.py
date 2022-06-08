import os
import sys
import math
import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from utils import TensorList
from models import siamrpn_alexnet, Anchors
from tracker.base_tracker import BaseTracker
from libs import sample_patch, sample_patch_multiscale
from features import GrayscaleFeature, ColornamesFeature, FhogFeature
from .dcf import (fft2, ifft2, resize_dft2, label_function_spatial, cosine_window, reg_window,
    feat_grid, resp_newton, norm_response, shift_sample, train_recf, ScaleFilter)

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
            
class ReSiam(BaseTracker):
    multiobj_mode = 'parallel'
    

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1  # 内部维护了一个属性，用于记录当前处于第几帧状态
        
        # 记录patch
        self.save_patch = True
        self.save_path_patch = '/root/autodl-tmp/uav123-person10-patch'
        if not os.path.exists(self.save_path_patch):
                os.mkdir(self.save_path_patch)
        self.save_patch_dpi = 300
        # 记录response
        self.save_response = True
        self.save_path_response = '/root/autodl-tmp/uav123-person10-response'
        if not os.path.exists(self.save_path_response):
                os.mkdir(self.save_path_response)
        self.save_response_dpi = 300
        
        
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        
        # Get position and size
        state = info['init_bbox']  # (x0,y0,w,h)
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2]).float()  # (cy, cx)
        self.target_sz = torch.Tensor([state[3], state[2]])  # (h, w)
        self.image_sz = torch.Tensor([image.shape[0], image.shape[1]])  #(H, W) of the image
        self.average_channel_value = np.mean(image, axis=(0, 1)).tolist()
        
        # Set the feature
        if self.params.feat.lower() == 'h':
            self.features = (FhogFeature(), GrayscaleFeature(), ColornamesFeature())
        elif 'd' in self.params.feat.lower():
            model = siamrpn_alexnet(self.params.use_alexnet_pad)
            try:
                # Absolute path
                model.load_state_dict(torch.load(self.params.net_path)['state_dict'], strict=True)
            except:
                # Relative path
                model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', '..', 'networks', self.params.net_path))['state_dict'], strict=True)
            if self.params.use_gpu:
                model.cuda()
            model.eval()
            if self.params.feat.lower() == 'hd':
                self.features = (FhogFeature(), GrayscaleFeature(), ColornamesFeature(), model)
            if self.params.feat.lower() == 'd':
                self.features = (model,)
        
        # Check if recf is used
        self.use_recf = self.params.get('use_recf', False)
        # Check if scale pyramids/scale filter/rpn head is used
        self.use_scale_pyramids = self.params.get('use_scale_pyramids', False)
        self.use_scale_filter = self.params.get('use_scale_filter', False)
        self.use_rpn_head = self.params.get('use_rpn_head', False)

        self.img_exemplar_sz = self.params.exemplar_side * torch.ones(2)
        self.update_target_scale()  # Initialize the target scale and base_target_sz
        
        # Essential params for each module
        if self.use_recf:
            self.init_feat_info()  # Initialize feature information
            self.init_recf()       # Initialize some learning things in recf: yf, cos_window, reg_window
        
        if self.use_scale_pyramids:
            self.num_scales = self.params.number_of_scales
            self.scale_step = self.params.scale_step
            scale_exp = torch.arange(-math.floor((self.num_scales - 1) / 2), math.ceil((self.num_scales - 1) / 2) + 1)
            self.scale_factors = self.scale_step ** scale_exp
            
        elif self.use_scale_filter:
            self.scale_filter = ScaleFilter(self.target_sz, self.average_channel_value, self.params)
            self.num_scales = self.scale_filter.num_scales
            self.scale_step = self.scale_filter.scale_step
            self.scale_factors = self.scale_filter.scale_factors
        
        elif self.use_rpn_head:
            self.anchor_num = len(self.params.anchor_ratios) * len(self.params.anchor_scales)
            self.scale_factors = torch.ones(1)
            
            if self.params.use_alexnet_pad:
                self.xcorr_output_side = int(self.features[-1].size(self.params.instance_side)[0])
            else:
                self.xcorr_output_side = (self.params.instance_side - self.params.exemplar_side) // self.params.anchor_stride + 1
            
            window = torch.hann_window(self.xcorr_output_side, periodic=False)[:, None].mm(
                     torch.hann_window(self.xcorr_output_side, periodic=False)[None, :])
            self.penalty_cos_window_flatten = window.flatten().repeat(self.anchor_num)
            self.penalty_cos_window = window
            self.anchors = self.generate_anchor(self.xcorr_output_side)  # dim=(所有anchor数,4)
            
            if self.params.enlarge_search_mode:
                if self.params.use_alexnet_pad:
                    self.xcorr_enlarge_output_side = int(self.features[-1].size(self.params.enlarge_search_side)[0])
                else:
                    self.xcorr_enlarge_output_side = (self.params.enlarge_search_side - self.params.exemplar_side) // self.params.anchor_stride + 1
                window_enlarge = torch.hann_window(self.xcorr_enlarge_output_side, periodic=False)[:, None].mm(
                                 torch.hann_window(self.xcorr_enlarge_output_side, periodic=False)[None, :])
                self.penalty_cos_window_enlarge_flatten = window_enlarge.flatten().repeat(self.anchor_num)
                self.penalty_cos_window_enlarge = window_enlarge
                self.anchors_enlarge = self.generate_anchor(self.xcorr_enlarge_output_side)  # dim=(所有anchor数,4)
        
        tic = time.time()
        #########################################################
        ################    INITIALIZE model    #################
        #########################################################
        
        # Extract the features within the corresponding search region
        sample_pos = self.pos.round()
        
        if self.use_recf:  # Using the first frame to train the ReCF
            im_patch = sample_patch(image, sample_pos, self.target_scale*self.img_sample_sz, self.img_sample_sz, self.average_channel_value)
            xl = TensorList([_f.extract(im_patch) for _f in self.features])
            self.update_recf(xl, init=True)
            
        if self.use_scale_filter:  # Initialize Scale Filter
            self.scale_filter.update(image, sample_pos, self.target_scale, self.base_target_sz, self.params)
        elif self.use_rpn_head:  # Build the template kernel
            im_patch = sample_patch(image, sample_pos, self.target_scale*self.img_exemplar_sz, self.img_exemplar_sz, self.average_channel_value)
            self.features[-1].get_exemplar_feat(im_patch)
            
            # 保存patch
            if self.save_patch:
                plt.imshow(im_patch[0].permute(1,2,0).type('torch.ByteTensor'))
                plt.axis('off')
                plt.savefig('{:s}/template_{:06d}.png'.format(self.save_path_patch, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_patch_dpi)
                plt.close()
            
        self.enlarge_search_mode = False
        
        elapsed_time = time.time() - tic
        self.time = []
        self.debug_info = {}
        self.time.append(elapsed_time)
        out = {'time': elapsed_time}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.frame_num += 1
        
        self.debug_info['frame_num'] = self.frame_num

        search_bbox = torch.cat((self.pos[[1,0]] - (self.img_sample_sz * self.target_scale-1)/2, 
                                self.img_sample_sz * self.target_scale))
        if self.params.get('use_alexnet_pad', True):
            receptive_bbox = torch.cat((self.pos[[1,0]] - (self.img_sample_sz * self.target_scale-1)/2, 
                                       self.img_sample_sz * self.target_scale))
        else:
            receptive_bbox = torch.cat((self.pos[[1,0]] - ((self.img_sample_sz-self.params.exemplar_side * torch.ones(2)) * self.target_scale-1)/2, 
                                       (self.img_sample_sz - self.params.exemplar_side * torch.ones(2)) * self.target_scale))
        
        tic = time.time()
        
        #########################################################
        ########    FEAT: Get feats of search region    #########
        #########################################################

        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.scale_factors
        # Extract features at multiple resolutions
        if len(sample_scales) == 1:
            im_patch = sample_patch(image, sample_pos, sample_scales[0]*self.img_sample_sz, self.img_sample_sz, self.average_channel_value)
        else:
            im_patch = sample_patch_multiscale(image, sample_pos, sample_scales, self.img_sample_sz, self.img_sample_sz, self.average_channel_value)
        xt = TensorList([_f.extract(im_patch) for _f in self.features])
        # 保存patch
        if self.save_patch:
            plt.imshow(im_patch[0].permute(1,2,0).type('torch.ByteTensor'))
            plt.axis('off')
            plt.savefig('{:s}/search_{:06d}.png'.format(self.save_path_patch, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_patch_dpi)
            plt.close()
        
        
        tic2 = time.time()
        
        #########################################################
        #########    DETECTION: Compute the response    #########
        #########################################################
        if self.use_recf and not self.enlarge_search_mode:
            cf_response, cf_response_f, xtf = self.apply_recf_detection(xt)
            
        if self.use_rpn_head:
            output_cls, output_loc = self.features[-1].track()
            cls_score_flatten, cls_score_map = self.convert_score(output_cls)
            if not self.enlarge_search_mode: 
                pred_bbox_flatten = self.convert_bbox(output_loc, self.anchors)
            else:
                pred_bbox_flatten = self.convert_bbox(output_loc, self.anchors_enlarge)

        tic3 = time.time()

        #########################################################
        ######    ESTIMATION: Get the pos/scale changes   #######
        #########################################################
        
        if self.use_recf and not self.use_rpn_head:
            translation_vec_feat, scale_idx, _, max_response = self.cf_localize_target(cf_response, cf_response_f)
            self.translation_vec_feat = translation_vec_feat
            estimation_flag = None
            
        elif self.use_rpn_head and not self.use_recf:
            pred_bbox_feat, cls_score_map, max_response = self.rpn_localize_target(cls_score_flatten, pred_bbox_flatten)
            self.translation_vec_feat = torch.Tensor([pred_bbox_feat[1], pred_bbox_feat[0]])
            
            if max_response < self.params.target_not_found_threshold:
                estimation_flag = 'not_found'
            else:
                estimation_flag = None
            
        elif self.use_recf and self.use_rpn_head:
            if not self.enlarge_search_mode:
                cf_response_centered_low_res = torch.fft.fftshift(cf_response, dim=(-2,-1)) \
                    if cf_response.shape[-2] == output_cls.shape[-2] and cf_response.shape[-1] == output_cls.shape[-1] \
                    else F.adaptive_avg_pool2d(torch.fft.fftshift(cf_response, dim=(-2,-1)), output_size=(output_cls.shape[-2], output_cls.shape[-1]))
                cf_response_centered_low_res = norm_response(cf_response_centered_low_res, 'relative_max')  # Make the max value equal to 1
                pred_bbox_feat, fused_response, max_response = self.localize_target(cls_score_map, 
                                                                                    pred_bbox_flatten, 
                                                                                    cf_response_centered_low_res)
            else:
                # Use rpnhead to realize re-detection when losing target
                pred_bbox_feat, cls_score_map, max_response = self.rpn_localize_target(cls_score_flatten, pred_bbox_flatten)
            
            self.translation_vec_feat = torch.tensor([pred_bbox_feat[1], pred_bbox_feat[0]])
            
            if max_response < self.params.target_not_found_threshold:
                estimation_flag = 'not_found'
            else:
                estimation_flag = None

        tic4 = time.time()
        
        #########################################################
        ##################    UPDATE Status    ##################
        #########################################################
        
        if estimation_flag not in ['not_found']:
            if self.use_recf and not self.use_rpn_head:
                translation_vec = self.translation_vec_feat * self.final_response_stride * \
                            self.target_scale * self.scale_factors[scale_idx]
            elif self.use_rpn_head:
                translation_vec = self.translation_vec_feat * self.target_scale
            
            pos = sample_pos + translation_vec
            self.update_pos(pos)

            if self.use_scale_pyramids:
                scale_change_factor = self.scale_factors[scale_idx].item()
                target_sz = self.base_target_sz * self.target_scale * scale_change_factor
                
            elif self.use_scale_filter:
                scale_change_factor = self.scale_filter.track(image, self.pos, self.target_scale, self.base_target_sz, self.params)
                target_sz = self.base_target_sz * self.target_scale * scale_change_factor
                
            elif self.use_rpn_head:
                target_w = self.target_sz[1] * (1 - self.smooth_size_lr) + \
                            pred_bbox_feat[2] * self.target_scale * self.smooth_size_lr
                target_h = self.target_sz[0] * (1 - self.smooth_size_lr) + \
                            pred_bbox_feat[3] * self.target_scale * self.smooth_size_lr
                target_sz = torch.tensor([target_h, target_w])

            self.update_target_sz(target_sz)
            self.update_target_scale()  # Update target scale and base_target_sz
            self.enlarge_search_mode = False
            
            # UPDATE Online Model
            if self.use_recf and self.frame_num % self.params.train_skipping == 0 and 'cf_response' in dir():
                sample_pos = self.pos.round()
                if self.params.use_detection_sample:
                    self.update_recf(xtf=xtf)  # Use detection sample to train filter
                else:
                    im_patch = sample_patch(image, sample_pos, self.target_scale*self.img_sample_sz, self.img_sample_sz, self.average_channel_value)
                    xl = TensorList([_f.extract(im_patch) for _f in self.features])
                    self.update_recf(xl=xl)
                    # 保存patch
                    if self.save_patch:
                        plt.imshow(im_patch[0].permute(1,2,0).type('torch.ByteTensor'))
                        plt.axis('off')
                        plt.savefig('{:s}/train_{:06d}.png'.format(self.save_path_patch, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_patch_dpi)
                        plt.close()
                    
            if self.use_scale_filter and self.frame_num % self.params.sf_train_skipping == 0:
                self.scale_filter.update(image, sample_pos, self.target_scale, self.base_target_sz, self.params)
        
        if estimation_flag == 'not_found' and self.params.enlarge_search_mode:
            if not self.enlarge_search_mode:
                self.update_target_scale(enlarge_mode=True)
                self.enlarge_search_mode = True
                
            if self.use_recf:
                self.update_recf(reset=True)
        
        # Return new state
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))  # 记录目标检测结果(x0,y0,w,h)
        
        toc = time.time()
        elapsed_time = toc - tic
        self.time.append(elapsed_time)
        out = {'target_bbox': new_state.tolist(), 
               'time':elapsed_time, 
               'search_bbox': search_bbox.tolist(), 
               'receptive_bbox': receptive_bbox.tolist(),
               'max_response': max_response}
        
        #########################################################
        ################    DEBUG infomation   ##################
        #########################################################
        # print("{:04d}: Curr.Elapsed.:{:.5f}s ( "
        #         "提取特征{:.5f}s +"
        #         "计算响应{:.5f}s + "
        #         "状态估计{:.5f} + "
        #         "状态更新{:.5f}, "
        #         "FPS:{:.2f}".format(self.frame_num, elapsed_time, 
        #                            tic2-tic, 
        #                            tic3-tic2, 
        #                            tic4-tic3, 
        #                            toc-tic4, 
        #                            self.frame_num/sum(self.time)))
        
        if self.save_response is True:
            # CF响应
            if 'cf_response' in dir():
                plt.imshow(torch.fft.fftshift(cf_response, dim=(-2,-1))[0])
                plt.axis('off')
                # plt.show()
                plt.savefig('{:s}/cf_response_{:06d}.png'.format(self.save_path_response, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_response_dpi)
                plt.close()
            if 'cls_score_map' in dir():
                # CLS响应
                max_score = torch.max(cls_score_map).item()
                _s_idx, _, _ = torch.where(cls_score_map==max_score)
                _s_idx = _s_idx[0]
                plt.imshow(cls_score_map[_s_idx].squeeze())
                plt.axis('off')
                # plt.show()
                plt.savefig('{:s}/cls_response_{:06d}.png'.format(self.save_path_response, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_response_dpi)
                plt.close()
            if 'fused_response' in dir():
                # 区域引导响应
                max_score = torch.max(fused_response).item()
                _s_idx, _, _ = torch.where(fused_response==max_score)
                _s_idx = _s_idx[0]
                plt.imshow(fused_response[_s_idx].squeeze())
                plt.axis('off')
                # plt.show()
                plt.savefig('{:s}/fused_response_{:06d}.png'.format(self.save_path_response, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_response_dpi)
                plt.close()
            if 'cf_response_centered_low_res' in dir():
                # norm_response响应
                plt.imshow(cf_response_centered_low_res.squeeze())
                plt.axis('off')
                # plt.show()
                plt.savefig('{:s}/cls_response_low_res_{:06d}.png'.format(self.save_path_response, self.frame_num), bbox_inches='tight',pad_inches = 0, dpi=self.save_response_dpi)
                plt.close()
        if self.visdom is not None:
            if 'cf_response' in dir():
                max_score = torch.max(cf_response).item()
                heatmap_opts = dict(
                    title='CF判别响应图(峰值:{:.4f})'.format(max_score),
                    marginleft=25,
                    marginright=0,
                    margintop=40,
                    marginbottom=30,
                )
                self.visdom.register(torch.fft.fftshift(cf_response, dim=(-2,-1)), 'heatmap', 1, 'CF判别响应图', opts=heatmap_opts)
                
            if 'cls_score_map' in dir():
                max_score = torch.max(cls_score_map).item()
                _s_idx, _, _ = torch.where(cls_score_map==max_score)
                _s_idx = _s_idx[0]
                display_cls_score_map = torch.cat(
                    [cls_score_map[i] if i == cls_score_map.shape[0]-1
                     else torch.cat([cls_score_map[i], torch.tensor(float('nan')).expand(cls_score_map[i].shape[0],2)], dim=1)
                     for i in range(cls_score_map.shape[0])], 
                    dim=1)
                heatmap_opts = dict(
                    title='RPN判别响应图(峰值:{:.2f})'.format(max_score),
                    marginleft=25,
                    marginright=0,
                    margintop=40,
                    marginbottom=30,
                    layoutopts=dict(
                        plotly=dict(
                            xaxis=dict(
                                tickmode='array',
                                tickvals=[int(cls_score_map.shape[1]/2) + i*(2 + int(cls_score_map.shape[1])) for i in range(cls_score_map.shape[0])],
                                ticktext=[f'来自锚{i+1}(峰值{torch.max(cls_score_map[i]).item():.5f})⭐' if i==_s_idx else f'来自锚{i+1}(峰值{torch.max(cls_score_map[i]).item():.5f})' for i in range(cls_score_map.shape[0])],
                                showgrid=False,
                                ),
                            yaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                )
                            )
                        )
                    )
                self.visdom.register(display_cls_score_map, 'heatmap', 1, 'RPN判别响应图', opts=heatmap_opts)
                
            if 'fused_response' in dir():
                max_score = torch.max(fused_response).item()
                _s_idx, _, _ = torch.where(fused_response==max_score)
                _s_idx = _s_idx[0]
                heatmap_opts = dict(
                    title='融合响应图(峰值:{:.2f})'.format(max_score),
                    marginleft=25,
                    marginright=0,
                    margintop=40,
                    marginbottom=30,
                )
                self.visdom.register(fused_response[_s_idx,:,:], 'heatmap', 1, '融合响应图', opts=heatmap_opts)
            
            if 'cf_response_centered_low_res' in dir():
                max_score_tmp = torch.max(cf_response_centered_low_res).item()
                heatmap_opts = dict(
                    title='CF响应图(裁剪后)(峰值:{:.2f})'.format(max_score_tmp),
                    marginleft=25,
                    marginright=0,
                    margintop=40,
                    marginbottom=30,
                )
                self.visdom.register(cf_response_centered_low_res, 'heatmap', 1, 'CF响应图(裁剪后)', opts=heatmap_opts)
                
                
            self.debug_info['目标框左上角点(x0,y0)'] = "({:.2f}, {:.2f})".format(new_state[0], new_state[1])
            self.debug_info['目标框宽高(w,h)'] = "({:.2f}, {:.2f})".format(new_state[2], new_state[3])
            self.debug_info['判别状态'] = estimation_flag
            self.debug_info['当前帧计算耗时(s)'] = "{:.3f}".format(elapsed_time)
            self.debug_info['平均跟踪速度(FPS)'] = self.frame_num/sum(self.time)
            self.visdom.register(self.debug_info, 'info_dict_curr', 1, '跟踪状态')
            
        return out
    
    def update_target_scale(self, enlarge_mode=False):
        """Use target_sz to update the target scale and base_target_sz
        """
        actual_exemplar_h = self.target_sz[0] + self.params.context_amount * torch.sum(self.target_sz)
        actual_exemplar_w = self.target_sz[1] + self.params.context_amount * torch.sum(self.target_sz)
        if not enlarge_mode:
            search_side = (self.params.instance_side / self.params.exemplar_side) * \
                        torch.sqrt(actual_exemplar_h * actual_exemplar_w)
            self.target_scale = math.sqrt(search_side ** 2 / self.params.instance_side ** 2)
        else:
            search_side = (self.params.enlarge_search_side / self.params.exemplar_side) * \
                        torch.sqrt(actual_exemplar_h * actual_exemplar_w)
            self.target_scale = math.sqrt(search_side ** 2 / self.params.enlarge_search_side ** 2)
        
        self.base_target_sz = self.target_sz / self.target_scale
        self.img_sample_sz = torch.round(search_side / self.target_scale) * torch.ones(2)
    
    def init_feat_info(self):
        '''Initialize feature information
        '''
        # Feature map size
        self.feature_sz = TensorList([_f.size(self.img_sample_sz) for _f in self.features])
        
        # Feature stride
        self.feature_stride = torch.Tensor([_f.cell_size for _f in self.features])
        
        # Fianl response size when fusing different resolution maps (considering the maximum resolution)
        min_stride_idx = torch.argmin(self.feature_stride)
        self.final_response_sz = self.feature_sz[min_stride_idx.item()]
        self.final_response_stride = self.feature_stride[min_stride_idx.item()]
        
        # To concatnate the features with the same feature sz
        self.feature_sz_gather = OrderedDict()
        for idx, feat_sz in enumerate(self.feature_sz.tolist()):
            if str(feat_sz) not in self.feature_sz_gather:
                self.feature_sz_gather[str(feat_sz)] = [idx]
            else:
                self.feature_sz_gather[str(feat_sz)].append(idx)
        self.feature_sz_unique = TensorList()
        for feat_sz in self.feature_sz_gather.keys():
            self.feature_sz_unique.append(torch.Tensor(eval(feat_sz)))
        
        self.feature_stride_gather = OrderedDict()
        for idx, feat_stride in enumerate(self.feature_stride.tolist()):
            if str(feat_stride) not in self.feature_stride_gather:
                self.feature_stride_gather[str(feat_stride)] = [idx]
            else:
                self.feature_stride_gather[str(feat_stride)].append(idx)
        self.feature_stride_unique = TensorList()
        for feat_stride in self.feature_stride_gather.keys():
            self.feature_stride_unique.append(torch.tensor(eval(feat_stride)))
    
    def init_recf(self):
        self.cos_window = cosine_window(self.feature_sz_unique) if not self.params.use_gpu \
                            else cosine_window(self.feature_sz_unique).cuda()
        self.update_label_function()
        if self.params.use_detection_sample:
            feature_grid_ky, feature_grid_kx = zip(*feat_grid(self.feature_sz_unique))
            self.feature_grid_ky = TensorList(feature_grid_ky) if not self.params.use_gpu else TensorList(feature_grid_ky).cuda()
            self.feature_grid_kx = TensorList(feature_grid_kx) if not self.params.use_gpu else TensorList(feature_grid_ky).cuda()
        if self.params.use_resp_newton:
            # For sub-grid optimization
            self.response_grid_ky, self.response_grid_kx = feat_grid(self.final_response_sz)
    
    def update_label_function(self, negative=False):
        _base_target_sz_feat = TensorList([(self.target_sz / self.target_scale) / feat_stride
                                           for feat_stride in self.feature_stride_unique])

        output_sigma = TensorList([torch.sqrt(torch.prod(torch.floor(base_feat_sz))) * \
                                   self.params.output_sigma_factor 
                                   for base_feat_sz in _base_target_sz_feat])
        
        y = label_function_spatial(self.feature_sz_unique, output_sigma, negative)
        self.yf = fft2(y)

        self.reg_window = reg_window(self.feature_sz_unique,
                                     _base_target_sz_feat, 
                                     self.params.reg_window_max, 
                                     self.params.reg_window_min,
                                     negative)
        
        if self.params.use_gpu:
            self.yf = self.yf.cuda()
            self.reg_window = self.reg_window.cuda()
    
    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True
    
    def generate_anchor(self, score_side):
        anchors = Anchors(self.params.anchor_stride,
                          self.params.anchor_ratios,
                          self.params.anchor_scales)
        anchor = anchors.anchors  # dim=(基anchor个数, 4)
        anchor_num = anchor.shape[0]
        
        x0, y0, x1, y1 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]  # (x0,y0,x1,y1)
        anchor = torch.stack([(x0+x1)*0.5, (y0+y1)*0.5, x1-x0, y1-y0], dim=1)  # 改成(cx,cy,w,h)
        
        anchor = anchor.repeat(1, score_side * score_side).reshape(-1, 4)  # 初始化所有anchor的(0,0,w,h)
        
        total_stride = anchors.stride
        ori = - (score_side // 2) * total_stride
        
        try:
            xx, yy = torch.meshgrid(torch.tensor([ori + total_stride * dx for dx in range(score_side)]), 
                                    torch.tensor([ori + total_stride * dy for dy in range(score_side)]), 
                                    indexing='xy')
        except:
            yy, xx = torch.meshgrid(torch.tensor([ori + total_stride * dy for dy in range(score_side)]),
                                    torch.tensor([ori + total_stride * dx for dx in range(score_side)]))
        
        xx, yy = xx.flatten().repeat(anchor_num), yy.flatten().repeat(anchor_num)  # 所有anchor的坐标值, 中心anchor的坐标为(0,0)
        anchor[:, 0], anchor[:, 1] = xx, yy
        
        return anchor
    
    def update_recf(self, xl=None, xtf=None, init=False, reset=False):
        """using the traning (learning) samples to train the ReCF
        """
        if not reset:
            if xl==None and xtf==None:
                raise ValueError('Need to use samples to train the recf. Now xl and xtf are both None.')
            
            # Train ReCF
            first_frame = not hasattr(self, 'model_xf')
            
            if self.params.use_detection_sample and not first_frame:
                # Use detection sample to train filter
                xlf = shift_sample(xtf, self.feature_grid_ky, self.feature_grid_kx, self.translation_vec_feat)
            else:
                xl = self.cat_feat(xl) if not self.params.use_gpu else self.cat_feat(xl).cuda()
                xlw = xl * self.cos_window
                xlf = fft2(xlw)
            
            if first_frame or init:
                self.xf_p = TensorList([torch.zeros_like(_xlf) for _xlf in xlf])  # previous sample
                self.wf_p = TensorList([torch.zeros_like(_xlf) for _xlf in xlf])  # previous filter
                self.model_xf = xlf
            else:
                self.xf_p = self.model_xf
                self.wf_p = self.wf
                # self.model_xf = (1 - self.params.learning_rate) * self.model_xf + self.params.learning_rate * xlf
                self.model_xf = TensorList([(1 - self.params.learning_rate) * _model_xf + self.params.learning_rate * _xlf for _model_xf, _xlf in zip(self.model_xf, xlf)])
            
            self.wf = train_recf(self.model_xf, self.yf, self.reg_window, self.xf_p, self.wf_p, self.params)
            
            if first_frame:
                self.wf_init = self.wf
        elif reset:
            self.wf = self.wf_init
    
    def apply_recf_detection(self, xt: TensorList):
        """xt: Testing sample features for detection.
        """
        xt = self.cat_feat(xt) if not self.params.use_gpu else self.cat_feat(xt).cuda()
        xtw = xt * self.cos_window
        xtf = fft2(xtw)
        
        # Compute the detection response
        response_f = self.wf.conj() * xtf
        response_f = TensorList([torch.sum(_rf, dim=1) for _rf in response_f])
        if len(self.feature_sz_unique) > 1:
            # Fuse reponse maps with different resolution
            response_f = torch.stack(resize_dft2(response_f, self.final_response_sz), dim=1)
            response_f = torch.sum(response_f, dim=1) if not self.params.use_gpu else torch.sum(response_f, dim=1).cpu()
        else:
            response_f = response_f[0] if not self.params.use_gpu else response_f[0].cpu()
        
        response = torch.real(ifft2(response_f))

        return response, response_f, xtf
    
    def cat_feat(self, x):
        """Concatenate the feature with the same resolution along with the channel dimension
        """
        x_gather = TensorList()
        for feat_idx in self.feature_sz_gather.values():
            x_gather.append(torch.cat(x[feat_idx], dim=1))
        return x_gather
    
    def convert_score(self, score):
        score_flatten = score.permute(1, 2, 3, 0).contiguous().view(2, -1)
        score_flatten = F.softmax(score_flatten, dim=0).data[1, :].cpu()
        
        score_map = score_flatten.view(-1, score.shape[-2], score.shape[-1])
        
        return score_flatten, score_map

    def convert_bbox(self, delta, anchor):
        delta_flatten = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu()

        delta_flatten[0, :] = delta_flatten[0, :] * anchor[:, 2] + anchor[:, 0]
        delta_flatten[1, :] = delta_flatten[1, :] * anchor[:, 3] + anchor[:, 1]
        delta_flatten[2, :] = torch.exp(delta_flatten[2, :]) * anchor[:, 2]
        delta_flatten[3, :] = torch.exp(delta_flatten[3, :]) * anchor[:, 3]
        
        # delta_map = delta_flatten.view(-1, delta.shape[-2], delta.shape[-1])
        
        return delta_flatten

    def cf_localize_target(self, response, response_f):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        if self.use_recf and self.params.use_resp_newton:
            # Use Newton method to refine the postition
            trans_row, trans_col, scale_idx, max_response = resp_newton(response, response_f,
                                                                        self.response_grid_ky, 
                                                                        self.response_grid_kx, 
                                                                        self.final_response_sz, 
                                                                        self.params.newton_iterations)
        else:
            # Find the maximum score and the corresponding position
            max_response = torch.max(response).item()
            scale_idx, row, col = torch.where(response==max_response)
            trans_row = (row + torch.floor((self.final_response_sz[0]-1)/2)) % self.final_response_sz[0] - \
                torch.floor((self.final_response_sz[0]-1)/2)
            trans_col = (col + torch.floor((self.final_response_sz[1]-1)/2)) % self.final_response_sz[1] - \
                torch.floor((self.final_response_sz[1]-1)/2)
        
        response_centered = torch.fft.fftshift(response, dim=(-2,-1))
        translation_vec_feat = torch.Tensor([trans_row, trans_col])
        
        return translation_vec_feat, scale_idx, response_centered, max_response
    
    def rpn_localize_target(self, cls_score_flatten, pred_bbox_flatten):
        # Scores after scale and ratio penaly
        sr_penalty_flatten = self.get_sr_penalty(pred_bbox_flatten)
        p_scores_flatten_init = sr_penalty_flatten * cls_score_flatten
        
        # Scores after cos_window penalty
        if not self.enlarge_search_mode:
            p_scores_flatten = (1 - self.params.penalty_cos_window_factor) * p_scores_flatten_init + \
                    self.params.penalty_cos_window_factor * self.penalty_cos_window_flatten
            # p_scores_flatten = p_scores_flatten_init * self.params.penalty_cos_window_factor * self.penalty_cos_window_flatten
            p_scores_map = p_scores_flatten.view(-1, self.xcorr_output_side, self.xcorr_output_side)
        else:
            p_scores_flatten = (1 - self.params.penalty_cos_window_factor) * p_scores_flatten_init + \
                    self.params.penalty_cos_window_factor * self.penalty_cos_window_enlarge_flatten
            p_scores_map = p_scores_flatten.view(-1, self.xcorr_enlarge_output_side, self.xcorr_enlarge_output_side)
                  
        best_idx = torch.argmax(p_scores_flatten)
        pred_bbox_feat = pred_bbox_flatten[:, best_idx]  # [cx, cy, w, h]
        
        # self.smooth_size_lr = self.params.smooth_size_lr
        self.smooth_size_lr = sr_penalty_flatten[best_idx] * cls_score_flatten[best_idx] * self.params.smooth_size_lr
        
        return pred_bbox_feat, p_scores_map, p_scores_flatten[best_idx]
    
    def get_sr_penalty(self, pred_bbox_flatten, return_map=False):
        # scale penalty and aspect ratio (w/h) penalty
        s_c = self.get_change_penalty(self.get_exemplar_side(self.params.context_amount, pred_bbox_flatten[2, :], pred_bbox_flatten[3, :]) /
                                          self.get_exemplar_side(self.params.context_amount, self.target_sz[1]/self.target_scale, self.target_sz[0]/self.target_scale))
        r_c = self.get_change_penalty((self.target_sz[1] / self.target_sz[0]) / 
                                          (pred_bbox_flatten[2, :] / pred_bbox_flatten[3, :]))
        sr_penalty_flatten = torch.exp(-r_c * s_c * self.params.penalty_factor)
        
        if return_map:
            if not self.enlarge_search_mode:
                sr_penalty_window = sr_penalty_flatten.view(-1, self.xcorr_output_side, self.xcorr_output_side)  # Debug
            else:
                sr_penalty_window = sr_penalty_flatten.view(-1, self.xcorr_enlarge_output_side, self.xcorr_enlarge_output_side)  # Debug
            return sr_penalty_flatten, sr_penalty_window
        else:
            return sr_penalty_flatten

    @staticmethod
    def get_exemplar_side(context_amount, target_w, target_h):
        padding = context_amount * (target_w + target_h)
        exemplar_side = torch.sqrt((target_w + padding) * (target_h + padding))
        return exemplar_side
    
    @staticmethod
    def get_change_penalty(change):
        return torch.maximum(change, 1 / change)
    
    def localize_target(self, cls_score_map, pred_bbox_flatten, cf_response):
        """
        args:
            response: Dim is (1, H, W)
        """
        # CF response window
        cf_response = (1 - self.params.penalty_cos_window_factor) * cf_response + \
                  self.params.penalty_cos_window_factor * self.penalty_cos_window
                  
        # Scores after scale and ratio penaly
        sr_penalty_flatten, sr_penalty_map = self.get_sr_penalty(pred_bbox_flatten, return_map=True)
        
        # Scores with response weights of CF
        fused_response = cls_score_map * cf_response * sr_penalty_map
        
        # Find the maximum score and the corresponding position
        max_response = torch.max(fused_response).item()
        best_idx = torch.argmax(fused_response.view(self.anchor_num, -1))
        
        pred_bbox_feat = pred_bbox_flatten[:, best_idx]
        
        self.smooth_size_lr = sr_penalty_flatten[best_idx] * max_response * self.params.smooth_size_lr
        
        return pred_bbox_feat, fused_response, max_response
    
    def update_pos(self, new_pos):
        if self.params.clamp_position:
            new_pos = torch.maximum(torch.Tensor([0, 0]), torch.minimum(self.image_sz, new_pos))
        self.pos = new_pos
    
    def update_target_sz(self, new_target_sz):
        if self.params.clamp_target_sz:
            new_target_sz = torch.maximum(torch.Tensor([2, 2]), torch.minimum(self.image_sz, new_target_sz))
        self.target_sz = new_target_sz
