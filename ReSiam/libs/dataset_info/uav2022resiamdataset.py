import numpy as np

from .data import Sequence, BaseDataset, SequenceList
from .load_text import load_text


class UAV2022ReSiamDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav2022resiam_path
        self.dset_name = 'UAV2022-ReSiam'
        
        self.sequence_info_list = self._get_sequence_info_list()        

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=[',', ' ', '\t'], dtype=np.float64, backend='numpy')
        
        return Sequence(sequence_info['name'], frames, self.dset_name, ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "basketballplayer1", "path": "data_seq/basketballplayer1/", "startFrame": 1, "endFrame": 1009, "nz": 6, "ext": "jpg", "anno_path": "anno/basketballplayer1.txt"},
            {"name": "basketballplayer2", "path": "data_seq/basketballplayer2/", "startFrame": 1, "endFrame": 676, "nz": 6, "ext": "jpg", "anno_path": "anno/basketballplayer2.txt"},
            {"name": "basketballplayer3", "path": "data_seq/basketballplayer3/", "startFrame": 1, "endFrame": 1180, "nz": 6, "ext": "jpg", "anno_path": "anno/basketballplayer3.txt"},
            {"name": "biker1", "path": "data_seq/biker1/", "startFrame": 1, "endFrame": 180, "nz": 6, "ext": "jpg", "anno_path": "anno/biker1.txt"},
            {"name": "biker2", "path": "data_seq/biker2/", "startFrame": 1, "endFrame": 327, "nz": 6, "ext": "jpg", "anno_path": "anno/biker2.txt"},
            {"name": "biker3", "path": "data_seq/biker3/", "startFrame": 1, "endFrame": 234, "nz": 6, "ext": "jpg", "anno_path": "anno/biker3.txt"},
            {"name": "biker4", "path": "data_seq/biker4/", "startFrame": 1, "endFrame": 840, "nz": 6, "ext": "jpg", "anno_path": "anno/biker4.txt"},
            {"name": "biker5", "path": "data_seq/biker5/", "startFrame": 1, "endFrame": 321, "nz": 6, "ext": "jpg", "anno_path": "anno/biker5.txt"},
            {"name": "biker6", "path": "data_seq/biker6/", "startFrame": 1, "endFrame": 301, "nz": 6, "ext": "jpg", "anno_path": "anno/biker6.txt"},
            {"name": "building_dark", "path": "data_seq/building_dark/", "startFrame": 1, "endFrame": 1536, "nz": 6, "ext": "jpg", "anno_path": "anno/building_dark.txt"},
            {"name": "car1", "path": "data_seq/car1/", "startFrame": 1, "endFrame": 159, "nz": 6, "ext": "jpg", "anno_path": "anno/car1.txt"},
            {"name": "car10", "path": "data_seq/car10/", "startFrame": 1, "endFrame": 254, "nz": 6, "ext": "jpg", "anno_path": "anno/car10.txt"},
            {"name": "car11_dark", "path": "data_seq/car11_dark/", "startFrame": 1, "endFrame": 528, "nz": 6, "ext": "jpg", "anno_path": "anno/car11_dark.txt"},
            {"name": "car12_dark", "path": "data_seq/car12_dark/", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg", "anno_path": "anno/car12_dark.txt"},
            {"name": "car13", "path": "data_seq/car13/", "startFrame": 1, "endFrame": 291, "nz": 6, "ext": "jpg", "anno_path": "anno/car13.txt"},
            {"name": "car14_dark", "path": "data_seq/car14_dark/", "startFrame": 1, "endFrame": 952, "nz": 6, "ext": "jpg", "anno_path": "anno/car14_dark.txt"},
            {"name": "car15", "path": "data_seq/car15/", "startFrame": 1, "endFrame": 100, "nz": 6, "ext": "jpg", "anno_path": "anno/car15.txt"},
            {"name": "car2", "path": "data_seq/car2/", "startFrame": 1, "endFrame": 171, "nz": 6, "ext": "jpg", "anno_path": "anno/car2.txt"},
            {"name": "car3_dark", "path": "data_seq/car3_dark/", "startFrame": 1, "endFrame": 2558, "nz": 6, "ext": "jpg", "anno_path": "anno/car3_dark.txt"},
            {"name": "car4", "path": "data_seq/car4/", "startFrame": 1, "endFrame": 286, "nz": 6, "ext": "jpg", "anno_path": "anno/car4.txt"},
            {"name": "car5", "path": "data_seq/car5/", "startFrame": 1, "endFrame": 221, "nz": 6, "ext": "jpg", "anno_path": "anno/car5.txt"},
            {"name": "car6", "path": "data_seq/car6/", "startFrame": 1, "endFrame": 183, "nz": 6, "ext": "jpg", "anno_path": "anno/car6.txt"},
            {"name": "car7", "path": "data_seq/car7/", "startFrame": 1, "endFrame": 315, "nz": 6, "ext": "jpg", "anno_path": "anno/car7.txt"},
            {"name": "car8", "path": "data_seq/car8/", "startFrame": 1, "endFrame": 184, "nz": 6, "ext": "jpg", "anno_path": "anno/car8.txt"},
            {"name": "car9_dark", "path": "data_seq/car9_dark/", "startFrame": 1, "endFrame": 529, "nz": 6, "ext": "jpg", "anno_path": "anno/car9_dark.txt"},
            {"name": "group1", "path": "data_seq/group1/", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg", "anno_path": "anno/group1.txt"},
            {"name": "group2_dark", "path": "data_seq/group2_dark/", "startFrame": 1, "endFrame": 609, "nz": 6, "ext": "jpg", "anno_path": "anno/group2_dark.txt"},
            {"name": "group3_dark", "path": "data_seq/group3_dark/", "startFrame": 1, "endFrame": 997, "nz": 6, "ext": "jpg", "anno_path": "anno/group3_dark.txt"},
            {"name": "group4", "path": "data_seq/group4/", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg", "anno_path": "anno/group4.txt"},
            {"name": "group5", "path": "data_seq/group5/", "startFrame": 1, "endFrame": 333, "nz": 6, "ext": "jpg", "anno_path": "anno/group5.txt"},
            {"name": "motor1", "path": "data_seq/motor1/", "startFrame": 1, "endFrame": 57, "nz": 6, "ext": "jpg", "anno_path": "anno/motor1.txt"},
            {"name": "motor2", "path": "data_seq/motor2/", "startFrame": 1, "endFrame": 53, "nz": 6, "ext": "jpg", "anno_path": "anno/motor2.txt"},
            {"name": "person1", "path": "data_seq/person1/", "startFrame": 1, "endFrame": 486, "nz": 6, "ext": "jpg", "anno_path": "anno/person1.txt"},
            {"name": "person2", "path": "data_seq/person2/", "startFrame": 1, "endFrame": 420, "nz": 6, "ext": "jpg", "anno_path": "anno/person2.txt"},
            {"name": "person3", "path": "data_seq/person3/", "startFrame": 1, "endFrame": 570, "nz": 6, "ext": "jpg", "anno_path": "anno/person3.txt"},
            {"name": "person4", "path": "data_seq/person4/", "startFrame": 1, "endFrame": 841, "nz": 6, "ext": "jpg", "anno_path": "anno/person4.txt"},
            {"name": "person5", "path": "data_seq/person5/", "startFrame": 1, "endFrame": 600, "nz": 6, "ext": "jpg", "anno_path": "anno/person5.txt"},
            {"name": "person6", "path": "data_seq/person6/", "startFrame": 1, "endFrame": 391, "nz": 6, "ext": "jpg", "anno_path": "anno/person6.txt"},
            {"name": "person7", "path": "data_seq/person7/", "startFrame": 1, "endFrame": 147, "nz": 6, "ext": "jpg", "anno_path": "anno/person7.txt"},
            {"name": "person8", "path": "data_seq/person8/", "startFrame": 1, "endFrame": 143, "nz": 6, "ext": "jpg", "anno_path": "anno/person8.txt"},
            {"name": "signpost_dark", "path": "data_seq/signpost_dark/", "startFrame": 1, "endFrame": 1074, "nz": 6, "ext": "jpg", "anno_path": "anno/signpost_dark.txt"},
            {"name": "swan", "path": "data_seq/swan/", "startFrame": 1, "endFrame": 300, "nz": 6, "ext": "jpg", "anno_path": "anno/swan.txt"},
        ]
        return sequence_info_list
