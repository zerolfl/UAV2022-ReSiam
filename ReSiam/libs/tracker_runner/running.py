import os
import sys
import time
import torch
import multiprocessing
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from collections import OrderedDict
from multiprocessing import Lock, Manager

from utils import ExperimentMail, istarmap
from libs.dataset_info.data import Sequence
from libs.tracker_runner.tracker import Tracker
from libs.dataset_info.load_text import load_text
from libs.processing.convert_res2mat import convert_res2mat
from libs.processing.extract_results import calc_seq_err_robust
from libs.processing.plot_results import get_auc_curve, get_prec_curve


def init_lock(l):
    global lock
    lock = l

def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, omit_init_time=False):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)
        
    # 保存到tracking_results/tracker, 要不然不同数据集出现同seq名就会出问题
    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    
    def save_bb(file, data):
        # tracked_bb = np.array(data).astype(int)
        # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
        tracked_bb = np.array(data).astype(np.float64)
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%f')

    def save_search_bb(file, data):
        # tracked_bb = np.array(data).astype(int)
        # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
        search_bb = np.array(data).astype(np.float64)
        np.savetxt(file, search_bb, delimiter=',', fmt='%f')
        
    def save_max_response(file, data):
        # tracked_bb = np.array(data).astype(int)
        # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')
        max_response = np.array(data).astype(np.float64)
        np.savetxt(file, max_response, delimiter=',', fmt='%f')
    
    def save_time(file, data):
        exec_times = np.array(data).astype(np.float64)
        np.savetxt(file, exec_times, delimiter=',', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        # receptive_bbox
        if key == 'search_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_search_bbox.txt'.format(base_results_path, obj_id)
                    save_search_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_search_bbox.txt'.format(base_results_path)
                save_search_bb(bbox_file, data)
                
        # receptive_bbox
        if key == 'max_response':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_max_response.txt'.format(base_results_path, obj_id)
                    save_max_response(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_max_response.txt'.format(base_results_path)
                save_max_response(bbox_file, data)
        
        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

def _gen_mat_res(seq, tracker, omit_init_time):
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    if tracker.experiment_name is None:
        if tracker.run_id is None:
            sot_res_mat_path = os.path.join(os.path.split(os.path.split(tracker.results_dir)[0])[0], 
                                        "matlab_res",
                                        "OPE_{}".format(seq.dataset), 
                                        "{}_{}/{}_{}_{}".format(tracker.name,tracker.parameter_name, 
                                                                seq.name, 
                                                                tracker.name, tracker.parameter_name))
        else:
            sot_res_mat_path = os.path.join(os.path.split(os.path.split(tracker.results_dir)[0])[0], 
                                        "matlab_res",
                                        "OPE_{}".format(seq.dataset), 
                                        "{}_{}_{:03d}/{}_{}_{}_{:03d}".format(tracker.name,tracker.parameter_name, tracker.run_id, 
                                                                              seq.name, 
                                                                              tracker.name, tracker.parameter_name, tracker.run_id, ))
    else:
        if tracker.run_id is None:
            sot_res_mat_path = os.path.join(os.path.split(os.path.split(tracker.results_dir)[0])[0], 
                                        "matlab_res-{}".format(tracker.experiment_name),
                                        "OPE_{}".format(seq.dataset), 
                                        "{}_{}/{}_{}_{}".format(tracker.name,tracker.parameter_name, 
                                                                seq.name, 
                                                                tracker.name, tracker.parameter_name))
        else:
            sot_res_mat_path = os.path.join(os.path.split(os.path.split(tracker.results_dir)[0])[0], 
                                        "matlab_res-{}".format(tracker.experiment_name),
                                        "OPE_{}".format(seq.dataset), 
                                        "{}_{}_{:03d}/{}_{}_{}_{:03d}".format(tracker.name,tracker.parameter_name, tracker.run_id, 
                                                                              seq.name, 
                                                                              tracker.name, tracker.parameter_name, tracker.run_id, ))
    
    if not os.path.exists(os.path.split(sot_res_mat_path)[0]):
        os.makedirs(os.path.split(sot_res_mat_path)[0])
        
    if (os.path.exists('{}.txt'.format(base_results_path)) and os.path.exists('{}_time.txt'.format(base_results_path))):
        """Convert seqname.txt and seqname_time.txt to seqname_trackername.mat file for benchmark evaluation"""
        convert_res2mat('{}.txt'.format(base_results_path),
                        '{}_time.txt'.format(base_results_path),
                        '{}.mat'.format(sot_res_mat_path),
                        omit_init_time=omit_init_time)
    else:
        print('Cannot generate mat results. Does not find {} or {}'.format('{}.txt'.format(base_results_path), '{}_time.txt'.format(base_results_path)))
    
def _disp_tracking_info(seq_idx, seq_num, seq_name, tracker_idx, tracker_num, tracker_name, param_name, run_id, hp_param, exec_time, fps):
    if hp_param is None:
        disp_info = 'Sequence ({:^3d}/{}): {:<18}| Tracker ({:^3d}/{}): {:<5} {:<15} {:<5}| Time: {:7.2f}s, FPS: {:7.2f}'.format(seq_idx, 
                                                                                            seq_num, 
                                                                                            seq_name, 
                                                                                            tracker_idx, 
                                                                                            tracker_num, 
                                                                                            tracker_name, 
                                                                                            param_name, 
                                                                                            str(run_id),
                                                                                            exec_time,
                                                                                            fps)
    else:
        disp_info = 'Sequence ({:^3d}/{}): {:<18}| Tracker ({:^3d}/{}): {:<5} {:<15} {:<5}| Time: {:7.2f}s, FPS: {:7.2f} | HP_Params: {}'.format(seq_idx, 
                                                                                            seq_num, 
                                                                                            seq_name, 
                                                                                            tracker_idx, 
                                                                                            tracker_num, 
                                                                                            tracker_name, 
                                                                                            param_name, 
                                                                                            str(run_id),
                                                                                            exec_time,
                                                                                            fps,
                                                                                            hp_param)
    return disp_info
    
def _init_eval_reports(send_frequency, dataset, trackers, plot_bin_gap=0.05):
    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0
    
    avg_overlap_all = float('nan') * torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    avg_fps_all = float('nan') * torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                    dtype=torch.float32)

    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)
    
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                     for t in trackers]
    
    eval_data = Manager().dict()
    eval_data.update({'sequences': seq_names, 
                 'trackers': tracker_names,
                 'valid_sequence': valid_sequence,
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap,
                 'ave_success_rate_plot_center': ave_success_rate_plot_center,
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm,
                 'avg_overlap_all': avg_overlap_all,
                 'avg_fps_all': avg_fps_all, 
                 'threshold_set_overlap': threshold_set_overlap,
                 'threshold_set_center': threshold_set_center,
                 'threshold_set_center_norm': threshold_set_center_norm,
                 'send_times': 0,
                 'send_frequency': send_frequency})
    
    return eval_data
        

def _eval_tracking_res(experiment_name, eval_data, seq, seq_id, tracker, trk_id, output: dict=None, omit_init_time=False):
    """Compute the prec. and succ.
    Args:
        compute_mode: 'once': compute evaluation results using the current tracker's output; 'file': compute by reading the res files.
    
    """
    
    anno_bb = torch.tensor(seq.ground_truth_rect)
    seq_length = anno_bb.shape[0]
    
    if output is None:  # 若存在结果则直接读取结果
        base_results_path = '{}/{}/{}'.format(tracker.results_dir, seq.dataset, seq.name)
        results_path = '{}.txt'.format(base_results_path)
        results_time_path  = '{}_time.txt'.format(base_results_path)
        if os.path.isfile(results_path):
            pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
            pred_time = torch.tensor(load_text(str(results_time_path), delimiter=('\t', ','), dtype=np.float64))
        else:
            print('Result not found. {}'.format(results_path))
            # raise Exception('Result not found. {}'.format(results_path))
    else:
        pred_bb = torch.tensor(output['target_bbox'])
        pred_time = torch.tensor(output['time'])

    # Calculate measures
    target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
    err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(pred_bb, anno_bb, seq.dataset, target_visible)
    
    eval_data['avg_fps_all'][seq_id, trk_id] = (seq_length - 1) / sum(pred_time[1:]) if omit_init_time else seq_length / sum(pred_time)
    eval_data['avg_overlap_all'][seq_id, trk_id] = err_overlap[valid_frame].mean()  # 平均每帧的Overlap
    eval_data['ave_success_rate_plot_overlap'][seq_id, trk_id, :] = (err_overlap.view(-1, 1) > eval_data['threshold_set_overlap'].view(1, -1)).sum(0).float() / seq_length  # Succ. series
    eval_data['ave_success_rate_plot_center'][seq_id, trk_id, :] = (err_center.view(-1, 1) <= eval_data['threshold_set_center'].view(1, -1)).sum(0).float() / seq_length  # Prec. series
    eval_data['ave_success_rate_plot_center_norm'][seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= eval_data['threshold_set_center_norm'].view(1, -1)).sum(0).float() / seq_length  # Norm. Prec. series
    
    complete_trkid = torch.where(torch.sum(torch.isnan(eval_data['avg_overlap_all']), 0)==0)[0]
    lock.acquire()  # 上锁写入读取数据
    if len(complete_trkid) == eval_data['avg_overlap_all'].shape[1] or \
        (len(complete_trkid) != 0 and len(complete_trkid) % (eval_data['send_frequency']*(eval_data['send_times']+1)) == 0):
        new_complete_trkid = complete_trkid.tolist()
        
        res_dict = dict()
        res_dict['Tracker ID'] = new_complete_trkid
        if eval_data['all_hp_dict'] != {}:
            for hp, hp_value in eval_data['all_hp_dict'].items():
                res_dict[hp] = [hp_value[i] for i in new_complete_trkid]
        
        # 计算不同数据集下的结果
        for dataset_name, seq_idx in eval_data['dataset_seq_idx'].items():
            # Success
            auc_curve, auc = get_auc_curve(eval_data['ave_success_rate_plot_overlap'][[[i,] for i in seq_idx], new_complete_trkid, :], torch.ones(len(seq_idx)).bool())
            res_dict['{} AUC'.format(dataset_name)] = list(map(lambda x:f'{x:.2f}', auc))
            # Precision
            prec_curve, prec_score = get_prec_curve(eval_data['ave_success_rate_plot_center'][[[i,] for i in seq_idx], new_complete_trkid, :], torch.ones(len(seq_idx)).bool())
            res_dict['{} DP'.format(dataset_name)] = list(map(lambda x:f'{x:.2f}', prec_score))
            # Norm Precision
            norm_prec_curve, norm_prec_score = get_prec_curve(eval_data['ave_success_rate_plot_center_norm'][[[i,] for i in seq_idx], new_complete_trkid, :], torch.ones(len(seq_idx)).bool())
            res_dict['{} Norm.DP'.format(dataset_name)] = list(map(lambda x:f'{x:.2f}', norm_prec_score))
            # FPS
            fps = eval_data['avg_fps_all'][[[i,] for i in seq_idx], new_complete_trkid].mean(0)
            res_dict['{} Avg.FPS'.format(dataset_name)] = list(map(lambda x:f'{x:.2f}', fps))
            
        # 计算总体结果
        # Success
        auc_curve, auc = get_auc_curve(eval_data['ave_success_rate_plot_overlap'][:, new_complete_trkid, :], eval_data['valid_sequence'].bool())
        res_dict['Overall AUC'] = list(map(lambda x:f'{x:.2f}', auc))
        # Precision
        prec_curve, prec_score = get_prec_curve(eval_data['ave_success_rate_plot_center'][:, new_complete_trkid, :], eval_data['valid_sequence'].bool())
        res_dict['Overall DP'] = list(map(lambda x:f'{x:.2f}', prec_score))
        # Norm Precision
        norm_prec_curve, norm_prec_score = get_prec_curve(eval_data['ave_success_rate_plot_center_norm'][:, new_complete_trkid, :], eval_data['valid_sequence'].bool())
        res_dict['Overall Norm.DP'] = list(map(lambda x:f'{x:.2f}', norm_prec_score))
        # FPS
        fps = eval_data['avg_fps_all'][:, new_complete_trkid].mean(0)
        res_dict['Avg.FPS'] = list(map(lambda x:f'{x:.2f}', fps))
        
        # 记录结果至CSV
        df = pd.DataFrame(res_dict)
        df = df.sort_values(by=['Overall AUC', 'Overall DP'], ascending=[False, False])
        _csv_file = '{}/{}.csv'.format(os.path.split(tracker.results_dir)[0], experiment_name)
        df.to_csv(_csv_file, index=None)
        
        if 'mail' in eval_data:
            mail = eval_data['mail']
            mail.note = mail.note.format(len(complete_trkid))
            mail.file_path = [_csv_file, tracker.default_param_file]
            mail.all_hyper_params_dict = tracker.all_hyper_params_dict
            mail.send()
        
        eval_data['send_times'] += 1
        
    lock.release()

def run_sequence(eval_data, experiment_name, mode,
                 seq: Sequence, tracker: Tracker, seq_idx=None, seq_num=None, tracker_idx=None, tracker_num=None, debug=False, visdom_info=None, omit_init_time=False):
    """Runs a tracker on a sequence."""
    
    def _results_exist():  # 判断是否有.txt结果文件存在
        bbox_file = '{}/{}/{}.txt'.format(tracker.results_dir, seq.dataset, seq.name)  # For otb, uav123, uavdt, dtb70, visdrone et al.
        return os.path.isfile(bbox_file)

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        _gen_mat_res(seq, tracker, omit_init_time)
        _eval_tracking_res(experiment_name, eval_data, seq, seq_idx-1, tracker, tracker_idx-1, omit_init_time=omit_init_time)
        disp_info = _disp_tracking_info(seq_idx, seq_num, seq.name, tracker_idx, tracker_num, tracker.name, tracker.parameter_name, str(tracker.run_id), tracker.hyper_params, -1, -1)
        lock.acquire()
        complete_num = torch.count_nonzero(~torch.isnan(eval_data['avg_overlap_all']))
        disp_info = '[{} ({}/{})] '.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), complete_num.item(), torch.numel(eval_data['avg_overlap_all'])) + disp_info
        if mode == 'parallel':
            print(disp_info)
            sys.stdout.flush()
        lock.release()
        return disp_info

    if debug:  # 只有使用debug才管可视化
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)  # 调用跟踪器类跑1个seq的方法
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    if omit_init_time:
        exec_time = sum(output['time'][1:])
        num_frames = len(output['time']) - 1
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    if not debug:
        _save_tracker_output(seq, tracker, output, omit_init_time)
        _gen_mat_res(seq, tracker, omit_init_time)
            
        # Evaluate tracking results. (Only supports single object mode)
        _eval_tracking_res(experiment_name, eval_data, seq, seq_idx-1, tracker, tracker_idx-1, output, omit_init_time)
    
    disp_info = _disp_tracking_info(seq_idx, seq_num, seq.name, tracker_idx, tracker_num, tracker.name, tracker.parameter_name, str(tracker.run_id), tracker.hyper_params, exec_time, num_frames / exec_time)
    lock.acquire()
    complete_num = torch.count_nonzero(~torch.isnan(eval_data['avg_overlap_all']))
    disp_info = '[{} ({}/{})] '.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), complete_num.item(), torch.numel(eval_data['avg_overlap_all'])) + disp_info
    if mode == 'parallel':
        print(disp_info)
        sys.stdout.flush()
    lock.release()
    return disp_info

def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None, omit_init_time=False, experiment_name=None, send_frequency=10, send_mail=False):
    """Runs a list of trackers on a dataset.
    args:
        # dataset: List of Sequence instances, forming a dataset.
        dataset: Dict of Sequence instances, forming a dataset. 改成了字典, 要不然不同数据集下有同名seq就有问题
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    dataset_info = {}
    dataset_seq_idx = {}
    for _seq_idx, seq in enumerate(dataset):
        dataset_info[seq.dataset] = dataset_info[seq.dataset] + 1 if seq.dataset in dataset_info else 1
        if seq.dataset not in dataset_seq_idx:
            dataset_seq_idx[seq.dataset] = [_seq_idx, ]
        else:
            dataset_seq_idx[seq.dataset].append(_seq_idx)
            
    disp_text = 'Evaluating {:4d} trackers on {:4d} sequences on {:2d} datasets ('.format(len(trackers), len(dataset), len(dataset_info))
    for dset_idx, dset_name in enumerate(dataset_info.keys()):
        disp_text = disp_text + "{}:{})".format(dset_name, dataset_info[dset_name]) if dset_idx == len(dataset_info)-1 \
                    else disp_text + "{}:{}, ".format(dset_name, dataset_info[dset_name])
    print(disp_text)

    visdom_info = {} if visdom_info is None else visdom_info
    
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'
        
    lock = Lock()
    eval_data = _init_eval_reports(send_frequency, dataset, trackers)
    all_hp_dict = {}
    for tracker in trackers:
        if tracker.hyper_params is not None:
            for hp, hp_value in tracker.hyper_params.items():
                all_hp_dict[hp] = [hp_value] if hp not in all_hp_dict else all_hp_dict[hp] + [hp_value]
    eval_data['all_hp_dict'] = all_hp_dict
    eval_data['dataset_seq_idx'] = dataset_seq_idx
    
    if experiment_name is None:
        experiment_name = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    
    param_list = [(eval_data, experiment_name, mode,) + 
                      (seq, 
                       tracker_info, 
                       ((all_seq_idx-1) % len(dataset) + 1), 
                       len(dataset), 
                       ((all_seq_idx-1) // len(dataset) + 1), 
                       len(trackers), 
                       debug, 
                       visdom_info,
                       omit_init_time) 
                      for all_seq_idx, (tracker_info, seq) in enumerate(product(trackers, dataset), start=1)]
    
    if send_mail:
        mail = ExperimentMail('{}'.format(experiment_name), '{}/'+str(len(trackers))+' Trackers')
        eval_data['mail'] = mail
    
    if mode == 'sequential':
        init_lock(lock)
        with tqdm(total=len(param_list), ncols=150, colour='CYAN') as pbar:
            for param in param_list:
                pbar.set_description('Experiment {}'.format(str(experiment_name)))
                _ = run_sequence(*param)
                tqdm.write(_)
                pbar.update(1)
                
    elif mode == 'parallel':
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=threads, initializer=init_lock, initargs=(lock,)) as pool:
            pool.starmap(run_sequence, param_list)
            # with tqdm(total=len(param_list), ncols=150, colour='CYAN') as pbar:
            #     for _ in pool.istarmap(run_sequence, param_list):
            #         pbar.set_description('Experiment {}'.format(str(experiment_name)))
            #         tqdm.write(_)
            #         pbar.update(1)

    print('Evaluation Done!')
