import sys
import numpy as np
from itertools import product
from libs import Tracker, trackerlist, get_dataset

def resiam_uav123():
    hyper_params = {
                    'penalty_cos_window_factor': [0.4],
                    'smooth_size_lr': [0.42],
                    }
    trackers = []
    hyper_params_comb = [dict(zip(hyper_params, v)) for v in product(*hyper_params.values())]
    experiment_name = sys._getframe().f_code.co_name
    for run_id, hp in enumerate(hyper_params_comb):
        trackers.append(Tracker('resiam', 'resiam', 
                                run_id=run_id, 
                                hyper_params=hp, 
                                experiment_name=experiment_name, 
                                all_hyper_params_dict=hyper_params))

    dataset = get_dataset('uav123')
    
    return trackers, dataset

def resiam_uav20l():
    hyper_params = {
                    'penalty_cos_window_factor': [0.42],
                    'smooth_size_lr': [0.48],
                    }
    trackers = []
    hyper_params_comb = [dict(zip(hyper_params, v)) for v in product(*hyper_params.values())]
    experiment_name = sys._getframe().f_code.co_name
    for run_id, hp in enumerate(hyper_params_comb):
        trackers.append(Tracker('resiam', 'resiam', 
                                run_id=run_id, 
                                hyper_params=hp, 
                                experiment_name=experiment_name, 
                                all_hyper_params_dict=hyper_params))

    dataset = get_dataset('uav20l')
    
    return trackers, dataset

def resiam_uav123_10fps():
    hyper_params = {
                    'penalty_cos_window_factor': [0.44],
                    'smooth_size_lr': [0.52],
                    }
    trackers = []
    hyper_params_comb = [dict(zip(hyper_params, v)) for v in product(*hyper_params.values())]
    experiment_name = sys._getframe().f_code.co_name
    for run_id, hp in enumerate(hyper_params_comb):
        trackers.append(Tracker('resiam', 'resiam', 
                                run_id=run_id, 
                                hyper_params=hp, 
                                experiment_name=experiment_name, 
                                all_hyper_params_dict=hyper_params))

    dataset = get_dataset('uav123_10fps')
    
    return trackers, dataset

def resiam_dtb70():
    hyper_params = {
                    'penalty_cos_window_factor': [0.44],
                    'smooth_size_lr': [0.46],
                    }
    trackers = []
    hyper_params_comb = [dict(zip(hyper_params, v)) for v in product(*hyper_params.values())]
    experiment_name = sys._getframe().f_code.co_name
    for run_id, hp in enumerate(hyper_params_comb):
        trackers.append(Tracker('resiam', 'resiam', 
                                run_id=run_id, 
                                hyper_params=hp, 
                                experiment_name=experiment_name, 
                                all_hyper_params_dict=hyper_params))

    dataset = get_dataset('dtb70')
    
    return trackers, dataset

def resiam_uavdt():
    hyper_params = {
                    'penalty_cos_window_factor': [0.44],
                    'smooth_size_lr': [0.48],
                    }
    trackers = []
    hyper_params_comb = [dict(zip(hyper_params, v)) for v in product(*hyper_params.values())]
    experiment_name = sys._getframe().f_code.co_name
    for run_id, hp in enumerate(hyper_params_comb):
        trackers.append(Tracker('resiam', 'resiam', 
                                run_id=run_id, 
                                hyper_params=hp, 
                                experiment_name=experiment_name, 
                                all_hyper_params_dict=hyper_params))

    dataset = get_dataset('uavdt')
    
    return trackers, dataset
