import os
import sys
import torch
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from libs import Tracker, trackerlist, get_dataset, run_dataset

torch.set_num_threads(1)

def run_tracker(tracker_name, tracker_param, run_id, dataset_name, sequence, debug, threads,
                visdom_info, omit_init_time):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]  # 实例化跟踪器模型

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info, omit_init_time=omit_init_time)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    # 测试UAV123/bike1
    parser.add_argument('--dataset_name', type=str, default='uav123', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).') 
    parser.add_argument('--sequence', type=str, default='bike1', help='Sequence number or name.')   # 'uav6'
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    parser.add_argument('--omit_init_time', action='store_true', help='Omit initialization time.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, 
                args.omit_init_time)


if __name__ == '__main__':
    main()
