import os
import sys
import numpy as np
import scipy.io as io

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from dataset_info.load_text import load_text


def convert_res2mat(res_txt, res_time_txt, mat_save_path, omit_init_time=False):
    bbox = load_text(res_txt, delimiter=('\t', ','), dtype=np.float64)
    elapsed_time = load_text(res_time_txt, delimiter=('\t', ','), dtype=np.float64)

    res_struct = {}
    res_struct['type'] = 'rect'
    res_struct['res'] = bbox
    
    if omit_init_time:
        res_struct['fps'] = (len(elapsed_time)-1) / sum(elapsed_time[1:])
    else:
        res_struct['fps'] = len(elapsed_time) / sum(elapsed_time)
        
    res_struct['len'] = len(bbox)
    res_struct['annoBegin'] = 1
    res_struct['startFrame'] = 1

    io.savemat(mat_save_path, {'results': [res_struct]})


if __name__ == "__main__":
    res_txt = r''
    res_time_txt = r''
    mat_save_path = r''
    omit_init_time = True
    
    convert_res2mat(res_txt, res_time_txt, mat_save_path, omit_init_time)