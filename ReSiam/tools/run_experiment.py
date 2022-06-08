import os
import sys
import argparse
import importlib
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from libs import run_dataset

torch.set_num_threads(1)

def run_experiment(experiment_module: str, experiment_name: str, debug: int, threads: int, omit_init_time: bool, send_frequency: int, send_mail: bool):
    """Run experiment.
    args:
        experiment_module: Name of experiment module in the experiments/ folder.
        experiment_name: Name of the experiment function.
        debug: Debug level.
        threads: Number of threads.
    """
    expr_module = importlib.import_module('experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset = expr_func()
    print('Running:  {}  {}'.format(experiment_module, experiment_name))
    run_dataset(dataset, trackers, debug, threads, omit_init_time=omit_init_time, experiment_name=experiment_name, send_frequency=send_frequency, send_mail=send_mail)


def main():
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument('--experiment_module', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment function.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--send_frequency', type=int, default=100, help='Frequency of recording results.')
    parser.add_argument('--omit_init_time', action='store_true', help='Omit initialization time.')
    parser.add_argument('--send_mail', action='store_true', help='Send email.')

    args = parser.parse_args()

    run_experiment(args.experiment_module, args.experiment_name, args.debug, args.threads, args.omit_init_time, args.send_frequency, args.send_mail)

if __name__ == '__main__':
    main()
