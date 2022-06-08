import numpy as np

from .data import Sequence, BaseDataset, SequenceList
from .load_text import load_text


class UAV123Dataset(BaseDataset):
    """ UAV123 dataset.

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf

    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self, version):
        super().__init__()
        self.version = version  # '30fps', '20l', '10fps'
        
        version_dset_name = {'30fps': 'UAV123', '20l':'UAV20L', '10fps':'UAV123@10fps'}
        version_path = {'30fps': self.env_settings.uav123_path, '20l':self.env_settings.uav20l_path, '10fps':self.env_settings.uav123_10fps_path}
        
        self.base_path = version_path[version]
        self.dset_name = version_dset_name[version]
        
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

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        # return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:],
        #                 object_class=sequence_info['object_class'])
        
        return Sequence(sequence_info['name'], frames, self.dset_name, ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        if self.version == '30fps':
            sequence_info_list = [
                {"name": "bike1", "path": "data_seq/bike1/", "startFrame": 1, "endFrame": 3085, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bike1.txt"},
                {"name": "bike2", "path": "data_seq/bike2/", "startFrame": 1, "endFrame": 553, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bike2.txt"},
                {"name": "bike3", "path": "data_seq/bike3/", "startFrame": 1, "endFrame": 433, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bike3.txt"},
                {"name": "bird1_1", "path": "data_seq/bird1/", "startFrame": 1, "endFrame": 253, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bird1_1.txt"},
                {"name": "bird1_2", "path": "data_seq/bird1/", "startFrame": 775, "endFrame": 1477, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bird1_2.txt"},
                {"name": "bird1_3", "path": "data_seq/bird1/", "startFrame": 1573, "endFrame": 2437, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/bird1_3.txt"},
                {"name": "boat1", "path": "data_seq/boat1/", "startFrame": 1, "endFrame": 901, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat1.txt"},
                {"name": "boat2", "path": "data_seq/boat2/", "startFrame": 1, "endFrame": 799, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat2.txt"},
                {"name": "boat3", "path": "data_seq/boat3/", "startFrame": 1, "endFrame": 901, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat3.txt"},
                {"name": "boat4", "path": "data_seq/boat4/", "startFrame": 1, "endFrame": 553, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat4.txt"},
                {"name": "boat5", "path": "data_seq/boat5/", "startFrame": 1, "endFrame": 505, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat5.txt"},
                {"name": "boat6", "path": "data_seq/boat6/", "startFrame": 1, "endFrame": 805, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat6.txt"},
                {"name": "boat7", "path": "data_seq/boat7/", "startFrame": 1, "endFrame": 535, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat7.txt"},
                {"name": "boat8", "path": "data_seq/boat8/", "startFrame": 1, "endFrame": 685, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat8.txt"},
                {"name": "boat9", "path": "data_seq/boat9/", "startFrame": 1, "endFrame": 1399, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/boat9.txt"},
                {"name": "building1", "path": "data_seq/building1/", "startFrame": 1, "endFrame": 469, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/building1.txt"},
                {"name": "building2", "path": "data_seq/building2/", "startFrame": 1, "endFrame": 577, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/building2.txt"},
                {"name": "building3", "path": "data_seq/building3/", "startFrame": 1, "endFrame": 829, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/building3.txt"},
                {"name": "building4", "path": "data_seq/building4/", "startFrame": 1, "endFrame": 787, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/building4.txt"},
                {"name": "building5", "path": "data_seq/building5/", "startFrame": 1, "endFrame": 481, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/building5.txt"},
                {"name": "car1_1", "path": "data_seq/car1/", "startFrame": 1, "endFrame": 751, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car1_1.txt"},
                {"name": "car1_2", "path": "data_seq/car1/", "startFrame": 751, "endFrame": 1627, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car1_2.txt"},
                {"name": "car1_3", "path": "data_seq/car1/", "startFrame": 1627, "endFrame": 2629, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car1_3.txt"},
                {"name": "car2", "path": "data_seq/car2/", "startFrame": 1, "endFrame": 1321, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car2.txt"},
                {"name": "car3", "path": "data_seq/car3/", "startFrame": 1, "endFrame": 1717, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car3.txt"},
                {"name": "car4", "path": "data_seq/car4/", "startFrame": 1, "endFrame": 1345, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car4.txt"},
                {"name": "car5", "path": "data_seq/car5/", "startFrame": 1, "endFrame": 745, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car5.txt"},
                {"name": "car6_1", "path": "data_seq/car6/", "startFrame": 1, "endFrame": 487, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car6_1.txt"},
                {"name": "car6_2", "path": "data_seq/car6/", "startFrame": 487, "endFrame": 1807, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car6_2.txt"},
                {"name": "car6_3", "path": "data_seq/car6/", "startFrame": 1807, "endFrame": 2953, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car6_3.txt"},
                {"name": "car6_4", "path": "data_seq/car6/", "startFrame": 2953, "endFrame": 3925, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car6_4.txt"},
                {"name": "car6_5", "path": "data_seq/car6/", "startFrame": 3925, "endFrame": 4861, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car6_5.txt"},
                {"name": "car7", "path": "data_seq/car7/", "startFrame": 1, "endFrame": 1033, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car7.txt"},
                {"name": "car8_1", "path": "data_seq/car8/", "startFrame": 1, "endFrame": 1357, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car8_1.txt"},
                {"name": "car8_2", "path": "data_seq/car8/", "startFrame": 1357, "endFrame": 2575, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car8_2.txt"},
                {"name": "car9", "path": "data_seq/car9/", "startFrame": 1, "endFrame": 1879, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car9.txt"},
                {"name": "car10", "path": "data_seq/car10/", "startFrame": 1, "endFrame": 1405, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car10.txt"},
                {"name": "car11", "path": "data_seq/car11/", "startFrame": 1, "endFrame": 337, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car11.txt"},
                {"name": "car12", "path": "data_seq/car12/", "startFrame": 1, "endFrame": 499, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car12.txt"},
                {"name": "car13", "path": "data_seq/car13/", "startFrame": 1, "endFrame": 415, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car13.txt"},
                {"name": "car14", "path": "data_seq/car14/", "startFrame": 1, "endFrame": 1327, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car14.txt"},
                {"name": "car15", "path": "data_seq/car15/", "startFrame": 1, "endFrame": 469, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car15.txt"},
                {"name": "car16_1", "path": "data_seq/car16/", "startFrame": 1, "endFrame": 415, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car16_1.txt"},
                {"name": "car16_2", "path": "data_seq/car16/", "startFrame": 415, "endFrame": 1993, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car16_2.txt"},
                {"name": "car17", "path": "data_seq/car17/", "startFrame": 1, "endFrame": 1057, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car17.txt"},
                {"name": "car18", "path": "data_seq/car18/", "startFrame": 1, "endFrame": 1207, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car18.txt"},
                {"name": "group1_1", "path": "data_seq/group1/", "startFrame": 1, "endFrame": 1333, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group1_1.txt"},
                {"name": "group1_2", "path": "data_seq/group1/", "startFrame": 1333, "endFrame": 2515, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group1_2.txt"},
                {"name": "group1_3", "path": "data_seq/group1/", "startFrame": 2515, "endFrame": 3925, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group1_3.txt"},
                {"name": "group1_4", "path": "data_seq/group1/", "startFrame": 3925, "endFrame": 4873, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group1_4.txt"},
                {"name": "group2_1", "path": "data_seq/group2/", "startFrame": 1, "endFrame": 907, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group2_1.txt"},
                {"name": "group2_2", "path": "data_seq/group2/", "startFrame": 907, "endFrame": 1771, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group2_2.txt"},
                {"name": "group2_3", "path": "data_seq/group2/", "startFrame": 1771, "endFrame": 2683, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group2_3.txt"},
                {"name": "group3_1", "path": "data_seq/group3/", "startFrame": 1, "endFrame": 1567, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group3_1.txt"},
                {"name": "group3_2", "path": "data_seq/group3/", "startFrame": 1567, "endFrame": 2827, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group3_2.txt"},
                {"name": "group3_3", "path": "data_seq/group3/", "startFrame": 2827, "endFrame": 4369, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group3_3.txt"},
                {"name": "group3_4", "path": "data_seq/group3/", "startFrame": 4369, "endFrame": 5527, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/group3_4.txt"},
                {"name": "person1", "path": "data_seq/person1/", "startFrame": 1, "endFrame": 799, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person1.txt"},
                {"name": "person2_1", "path": "data_seq/person2/", "startFrame": 1, "endFrame": 1189, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person2_1.txt"},
                {"name": "person2_2", "path": "data_seq/person2/", "startFrame": 1189, "endFrame": 2623, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person2_2.txt"},
                {"name": "person3", "path": "data_seq/person3/", "startFrame": 1, "endFrame": 643, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person3.txt"},
                {"name": "person4_1", "path": "data_seq/person4/", "startFrame": 1, "endFrame": 1501, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person4_1.txt"},
                {"name": "person4_2", "path": "data_seq/person4/", "startFrame": 1501, "endFrame": 2743, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person4_2.txt"},
                {"name": "person5_1", "path": "data_seq/person5/", "startFrame": 1, "endFrame": 877, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person5_1.txt"},
                {"name": "person5_2", "path": "data_seq/person5/", "startFrame": 877, "endFrame": 2101, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person5_2.txt"},
                {"name": "person6", "path": "data_seq/person6/", "startFrame": 1, "endFrame": 901, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person6.txt"},
                {"name": "person7_1", "path": "data_seq/person7/", "startFrame": 1, "endFrame": 1249, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person7_1.txt"},
                {"name": "person7_2", "path": "data_seq/person7/", "startFrame": 1249, "endFrame": 2065, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person7_2.txt"},
                {"name": "person8_1", "path": "data_seq/person8/", "startFrame": 1, "endFrame": 1075, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person8_1.txt"},
                {"name": "person8_2", "path": "data_seq/person8/", "startFrame": 1075, "endFrame": 1525, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person8_2.txt"},
                {"name": "person9", "path": "data_seq/person9/", "startFrame": 1, "endFrame": 661, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person9.txt"},
                {"name": "person10", "path": "data_seq/person10/", "startFrame": 1, "endFrame": 1021, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person10.txt"},
                {"name": "person11", "path": "data_seq/person11/", "startFrame": 1, "endFrame": 721, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person11.txt"},
                {"name": "person12_1", "path": "data_seq/person12/", "startFrame": 1, "endFrame": 601, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person12_1.txt"},
                {"name": "person12_2", "path": "data_seq/person12/", "startFrame": 601, "endFrame": 1621, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person12_2.txt"},
                {"name": "person13", "path": "data_seq/person13/", "startFrame": 1, "endFrame": 883, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person13.txt"},
                {"name": "person14_1", "path": "data_seq/person14/", "startFrame": 1, "endFrame": 847, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_1.txt"},
                {"name": "person14_2", "path": "data_seq/person14/", "startFrame": 847, "endFrame": 1813, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_2.txt"},
                {"name": "person14_3", "path": "data_seq/person14/", "startFrame": 1813, "endFrame": 2923, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_3.txt"},
                {"name": "person15", "path": "data_seq/person15/", "startFrame": 1, "endFrame": 1339, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person15.txt"},
                {"name": "person16", "path": "data_seq/person16/", "startFrame": 1, "endFrame": 1147, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person16.txt"},
                {"name": "person17_1", "path": "data_seq/person17/", "startFrame": 1, "endFrame": 1501, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_1.txt"},
                {"name": "person17_2", "path": "data_seq/person17/", "startFrame": 1501, "endFrame": 2347, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_2.txt"},
                {"name": "person18", "path": "data_seq/person18/", "startFrame": 1, "endFrame": 1393, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person18.txt"},
                {"name": "person19_1", "path": "data_seq/person19/", "startFrame": 1, "endFrame": 1243, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_1.txt"},
                {"name": "person19_2", "path": "data_seq/person19/", "startFrame": 1243, "endFrame": 2791, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_2.txt"},
                {"name": "person19_3", "path": "data_seq/person19/", "startFrame": 2791, "endFrame": 4357, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_3.txt"},
                {"name": "person20", "path": "data_seq/person20/", "startFrame": 1, "endFrame": 1783, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person20.txt"},
                {"name": "person21", "path": "data_seq/person21/", "startFrame": 1, "endFrame": 487, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person21.txt"},
                {"name": "person22", "path": "data_seq/person22/", "startFrame": 1, "endFrame": 199, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person22.txt"},
                {"name": "person23", "path": "data_seq/person23/", "startFrame": 1, "endFrame": 397, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person23.txt"},
                {"name": "truck1", "path": "data_seq/truck1/", "startFrame": 1, "endFrame": 463, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/truck1.txt"},
                {"name": "truck2", "path": "data_seq/truck2/", "startFrame": 1, "endFrame": 385, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/truck2.txt"},
                {"name": "truck3", "path": "data_seq/truck3/", "startFrame": 1, "endFrame": 535, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/truck3.txt"},
                {"name": "truck4_1", "path": "data_seq/truck4/", "startFrame": 1, "endFrame": 577, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/truck4_1.txt"},
                {"name": "truck4_2", "path": "data_seq/truck4/", "startFrame": 577, "endFrame": 1261, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/truck4_2.txt"},
                {"name": "uav1_1", "path": "data_seq/uav1/", "startFrame": 1, "endFrame": 1555, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav1_1.txt"},
                {"name": "uav1_2", "path": "data_seq/uav1/", "startFrame": 1555, "endFrame": 2377, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav1_2.txt"},
                {"name": "uav1_3", "path": "data_seq/uav1/", "startFrame": 2473, "endFrame": 3469, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav1_3.txt"},
                {"name": "uav2", "path": "data_seq/uav2/", "startFrame": 1, "endFrame": 133, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav2.txt"},
                {"name": "uav3", "path": "data_seq/uav3/", "startFrame": 1, "endFrame": 265, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav3.txt"},
                {"name": "uav4", "path": "data_seq/uav4/", "startFrame": 1, "endFrame": 157, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav4.txt"},
                {"name": "uav5", "path": "data_seq/uav5/", "startFrame": 1, "endFrame": 139, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav5.txt"},
                {"name": "uav6", "path": "data_seq/uav6/", "startFrame": 1, "endFrame": 109, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav6.txt"},
                {"name": "uav7", "path": "data_seq/uav7/", "startFrame": 1, "endFrame": 373, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav7.txt"},
                {"name": "uav8", "path": "data_seq/uav8/", "startFrame": 1, "endFrame": 301, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/uav8.txt"},
                {"name": "wakeboard1", "path": "data_seq/wakeboard1/", "startFrame": 1, "endFrame": 421, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard1.txt"},
                {"name": "wakeboard2", "path": "data_seq/wakeboard2/", "startFrame": 1, "endFrame": 733, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard2.txt"},
                {"name": "wakeboard3", "path": "data_seq/wakeboard3/", "startFrame": 1, "endFrame": 823, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard3.txt"},
                {"name": "wakeboard4", "path": "data_seq/wakeboard4/", "startFrame": 1, "endFrame": 697, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard4.txt"},
                {"name": "wakeboard5", "path": "data_seq/wakeboard5/", "startFrame": 1, "endFrame": 1675, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard5.txt"},
                {"name": "wakeboard6", "path": "data_seq/wakeboard6/", "startFrame": 1, "endFrame": 1165, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard6.txt"},
                {"name": "wakeboard7", "path": "data_seq/wakeboard7/", "startFrame": 1, "endFrame": 199, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard7.txt"},
                {"name": "wakeboard8", "path": "data_seq/wakeboard8/", "startFrame": 1, "endFrame": 1543, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard8.txt"},
                {"name": "wakeboard9", "path": "data_seq/wakeboard9/", "startFrame": 1, "endFrame": 355, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard9.txt"},
                {"name": "wakeboard10", "path": "data_seq/wakeboard10/", "startFrame": 1, "endFrame": 469, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard10.txt"},
                {"name": "car1_s", "path": "data_seq/car1_s/", "startFrame": 1, "endFrame": 1475, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car1_s.txt"},
                {"name": "car2_s", "path": "data_seq/car2_s/", "startFrame": 1, "endFrame": 320, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car2_s.txt"},
                {"name": "car3_s", "path": "data_seq/car3_s/", "startFrame": 1, "endFrame": 1300, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car3_s.txt"},
                {"name": "car4_s", "path": "data_seq/car4_s/", "startFrame": 1, "endFrame": 830, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/car4_s.txt"},
                {"name": "person1_s", "path": "data_seq/person1_s/", "startFrame": 1, "endFrame": 1600, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person1_s.txt"},
                {"name": "person2_s", "path": "data_seq/person2_s/", "startFrame": 1, "endFrame": 250, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person2_s.txt"},
                {"name": "person3_s", "path": "data_seq/person3_s/", "startFrame": 1, "endFrame": 505, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person3_s.txt"},
            ]
            
        elif self.version == '20l':
            sequence_info_list = [
                {"name": "bike1", "path": "data_seq/bike1/", "startFrame": 1, "endFrame": 3085, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/bike1.txt"},
                {"name": "bird1", "path": "data_seq/bird1/", "startFrame": 1, "endFrame": 2437, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/bird1.txt"},
                {"name": "car1", "path": "data_seq/car1/", "startFrame": 1, "endFrame": 2629, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car1.txt"},
                {"name": "car3", "path": "data_seq/car3/", "startFrame": 1, "endFrame": 1717, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car3.txt"},
                {"name": "car6", "path": "data_seq/car6/", "startFrame": 1, "endFrame": 4861, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car6.txt"},
                {"name": "car8", "path": "data_seq/car8/", "startFrame": 1, "endFrame": 2575, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car8.txt"},
                {"name": "car9", "path": "data_seq/car9/", "startFrame": 1, "endFrame": 1879, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car9.txt"},
                {"name": "car16", "path": "data_seq/car16/", "startFrame": 1, "endFrame": 1993, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/car16.txt"},
                {"name": "group1", "path": "data_seq/group1/", "startFrame": 1, "endFrame": 4873, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/group1.txt"},
                {"name": "group2", "path": "data_seq/group2/", "startFrame": 1, "endFrame": 2683, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/group2.txt"},
                {"name": "group3", "path": "data_seq/group3/", "startFrame": 1, "endFrame": 5527, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/group3.txt"},
                {"name": "person2", "path": "data_seq/person2/", "startFrame": 1, "endFrame": 2623, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person2.txt"},
                {"name": "person4", "path": "data_seq/person4/", "startFrame": 1, "endFrame": 2743, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person4.txt"},
                {"name": "person5", "path": "data_seq/person5/", "startFrame": 1, "endFrame": 2101, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person5.txt"},
                {"name": "person7", "path": "data_seq/person7/", "startFrame": 1, "endFrame": 2065, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person7.txt"},
                {"name": "person14", "path": "data_seq/person14/", "startFrame": 1, "endFrame": 2923, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person14.txt"},
                {"name": "person17", "path": "data_seq/person17/", "startFrame": 1, "endFrame": 2347, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person17.txt"},
                {"name": "person19", "path": "data_seq/person19/", "startFrame": 1, "endFrame": 4357, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person19.txt"},
                {"name": "person20", "path": "data_seq/person20/", "startFrame": 1, "endFrame": 1783, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/person20.txt"},
                {"name": "uav1", "path": "data_seq/uav1/", "startFrame": 1, "endFrame": 3469, "nz": 6, "ext": "jpg", "anno_path": "anno/UAV20L/uav1.txt"},
            ]
            
        elif self.version == '10fps':
            sequence_info_list = [
                {"name": "bike1", "path": "data_seq/bike1/", "startFrame": 1, "endFrame": 1029, "nz": 6, "ext": "jpg", "anno_path": "anno/bike1.txt"},
                {"name": "bike2", "path": "data_seq/bike2/", "startFrame": 1, "endFrame": 185, "nz": 6, "ext": "jpg", "anno_path": "anno/bike2.txt"},
                {"name": "bike3", "path": "data_seq/bike3/", "startFrame": 1, "endFrame": 145, "nz": 6, "ext": "jpg", "anno_path": "anno/bike3.txt"},
                {"name": "bird1_1", "path": "data_seq/bird1/", "startFrame": 1, "endFrame": 85, "nz": 6, "ext": "jpg", "anno_path": "anno/bird1_1.txt"},
                {"name": "bird1_2", "path": "data_seq/bird1/", "startFrame": 259, "endFrame": 493, "nz": 6, "ext": "jpg", "anno_path": "anno/bird1_2.txt"},
                {"name": "bird1_3", "path": "data_seq/bird1/", "startFrame": 525, "endFrame": 813, "nz": 6, "ext": "jpg", "anno_path": "anno/bird1_3.txt"},
                {"name": "boat1", "path": "data_seq/boat1/", "startFrame": 1, "endFrame": 301, "nz": 6, "ext": "jpg", "anno_path": "anno/boat1.txt"},
                {"name": "boat2", "path": "data_seq/boat2/", "startFrame": 1, "endFrame": 267, "nz": 6, "ext": "jpg", "anno_path": "anno/boat2.txt"},
                {"name": "boat3", "path": "data_seq/boat3/", "startFrame": 1, "endFrame": 301, "nz": 6, "ext": "jpg", "anno_path": "anno/boat3.txt"},
                {"name": "boat4", "path": "data_seq/boat4/", "startFrame": 1, "endFrame": 185, "nz": 6, "ext": "jpg", "anno_path": "anno/boat4.txt"},
                {"name": "boat5", "path": "data_seq/boat5/", "startFrame": 1, "endFrame": 169, "nz": 6, "ext": "jpg", "anno_path": "anno/boat5.txt"},
                {"name": "boat6", "path": "data_seq/boat6/", "startFrame": 1, "endFrame": 269, "nz": 6, "ext": "jpg", "anno_path": "anno/boat6.txt"},
                {"name": "boat7", "path": "data_seq/boat7/", "startFrame": 1, "endFrame": 179, "nz": 6, "ext": "jpg", "anno_path": "anno/boat7.txt"},
                {"name": "boat8", "path": "data_seq/boat8/", "startFrame": 1, "endFrame": 229, "nz": 6, "ext": "jpg", "anno_path": "anno/boat8.txt"},
                {"name": "boat9", "path": "data_seq/boat9/", "startFrame": 1, "endFrame": 467, "nz": 6, "ext": "jpg", "anno_path": "anno/boat9.txt"},
                {"name": "building1", "path": "data_seq/building1/", "startFrame": 1, "endFrame": 157, "nz": 6, "ext": "jpg", "anno_path": "anno/building1.txt"},
                {"name": "building2", "path": "data_seq/building2/", "startFrame": 1, "endFrame": 193, "nz": 6, "ext": "jpg", "anno_path": "anno/building2.txt"},
                {"name": "building3", "path": "data_seq/building3/", "startFrame": 1, "endFrame": 277, "nz": 6, "ext": "jpg", "anno_path": "anno/building3.txt"},
                {"name": "building4", "path": "data_seq/building4/", "startFrame": 1, "endFrame": 263, "nz": 6, "ext": "jpg", "anno_path": "anno/building4.txt"},
                {"name": "building5", "path": "data_seq/building5/", "startFrame": 1, "endFrame": 161, "nz": 6, "ext": "jpg", "anno_path": "anno/building5.txt"},
                {"name": "car1_1", "path": "data_seq/car1/", "startFrame": 1, "endFrame": 251, "nz": 6, "ext": "jpg", "anno_path": "anno/car1_1.txt"},
                {"name": "car1_2", "path": "data_seq/car1/", "startFrame": 251, "endFrame": 543, "nz": 6, "ext": "jpg", "anno_path": "anno/car1_2.txt"},
                {"name": "car1_3", "path": "data_seq/car1/", "startFrame": 543, "endFrame": 877, "nz": 6, "ext": "jpg", "anno_path": "anno/car1_3.txt"},
                {"name": "car2", "path": "data_seq/car2/", "startFrame": 1, "endFrame": 441, "nz": 6, "ext": "jpg", "anno_path": "anno/car2.txt"},
                {"name": "car3", "path": "data_seq/car3/", "startFrame": 1, "endFrame": 573, "nz": 6, "ext": "jpg", "anno_path": "anno/car3.txt"},
                {"name": "car4", "path": "data_seq/car4/", "startFrame": 1, "endFrame": 449, "nz": 6, "ext": "jpg", "anno_path": "anno/car4.txt"},
                {"name": "car5", "path": "data_seq/car5/", "startFrame": 1, "endFrame": 249, "nz": 6, "ext": "jpg", "anno_path": "anno/car5.txt"},
                {"name": "car6_1", "path": "data_seq/car6/", "startFrame": 1, "endFrame": 163, "nz": 6, "ext": "jpg", "anno_path": "anno/car6_1.txt"},
                {"name": "car6_2", "path": "data_seq/car6/", "startFrame": 163, "endFrame": 603, "nz": 6, "ext": "jpg", "anno_path": "anno/car6_2.txt"},
                {"name": "car6_3", "path": "data_seq/car6/", "startFrame": 603, "endFrame": 985, "nz": 6, "ext": "jpg", "anno_path": "anno/car6_3.txt"},
                {"name": "car6_4", "path": "data_seq/car6/", "startFrame": 985, "endFrame": 1309, "nz": 6, "ext": "jpg", "anno_path": "anno/car6_4.txt"},
                {"name": "car6_5", "path": "data_seq/car6/", "startFrame": 1309, "endFrame": 1621, "nz": 6, "ext": "jpg", "anno_path": "anno/car6_5.txt"},
                {"name": "car7", "path": "data_seq/car7/", "startFrame": 1, "endFrame": 345, "nz": 6, "ext": "jpg", "anno_path": "anno/car7.txt"},
                {"name": "car8_1", "path": "data_seq/car8/", "startFrame": 1, "endFrame": 453, "nz": 6, "ext": "jpg", "anno_path": "anno/car8_1.txt"},
                {"name": "car8_2", "path": "data_seq/car8/", "startFrame": 453, "endFrame": 859, "nz": 6, "ext": "jpg", "anno_path": "anno/car8_2.txt"},
                {"name": "car9", "path": "data_seq/car9/", "startFrame": 1, "endFrame": 627, "nz": 6, "ext": "jpg", "anno_path": "anno/car9.txt"},
                {"name": "car10", "path": "data_seq/car10/", "startFrame": 1, "endFrame": 469, "nz": 6, "ext": "jpg", "anno_path": "anno/car10.txt"},
                {"name": "car11", "path": "data_seq/car11/", "startFrame": 1, "endFrame": 113, "nz": 6, "ext": "jpg", "anno_path": "anno/car11.txt"},
                {"name": "car12", "path": "data_seq/car12/", "startFrame": 1, "endFrame": 167, "nz": 6, "ext": "jpg", "anno_path": "anno/car12.txt"},
                {"name": "car13", "path": "data_seq/car13/", "startFrame": 1, "endFrame": 139, "nz": 6, "ext": "jpg", "anno_path": "anno/car13.txt"},
                {"name": "car14", "path": "data_seq/car14/", "startFrame": 1, "endFrame": 443, "nz": 6, "ext": "jpg", "anno_path": "anno/car14.txt"},
                {"name": "car15", "path": "data_seq/car15/", "startFrame": 1, "endFrame": 157, "nz": 6, "ext": "jpg", "anno_path": "anno/car15.txt"},
                {"name": "car16_1", "path": "data_seq/car16/", "startFrame": 1, "endFrame": 139, "nz": 6, "ext": "jpg", "anno_path": "anno/car16_1.txt"},
                {"name": "car16_2", "path": "data_seq/car16/", "startFrame": 139, "endFrame": 665, "nz": 6, "ext": "jpg", "anno_path": "anno/car16_2.txt"},
                {"name": "car17", "path": "data_seq/car17/", "startFrame": 1, "endFrame": 353, "nz": 6, "ext": "jpg", "anno_path": "anno/car17.txt"},
                {"name": "car18", "path": "data_seq/car18/", "startFrame": 1, "endFrame": 403, "nz": 6, "ext": "jpg", "anno_path": "anno/car18.txt"},
                {"name": "group1_1", "path": "data_seq/group1/", "startFrame": 1, "endFrame": 445, "nz": 6, "ext": "jpg", "anno_path": "anno/group1_1.txt"},
                {"name": "group1_2", "path": "data_seq/group1/", "startFrame": 445, "endFrame": 839, "nz": 6, "ext": "jpg", "anno_path": "anno/group1_2.txt"},
                {"name": "group1_3", "path": "data_seq/group1/", "startFrame": 839, "endFrame": 1309, "nz": 6, "ext": "jpg", "anno_path": "anno/group1_3.txt"},
                {"name": "group1_4", "path": "data_seq/group1/", "startFrame": 1309, "endFrame": 1625, "nz": 6, "ext": "jpg", "anno_path": "anno/group1_4.txt"},
                {"name": "group2_1", "path": "data_seq/group2/", "startFrame": 1, "endFrame": 303, "nz": 6, "ext": "jpg", "anno_path": "anno/group2_1.txt"},
                {"name": "group2_2", "path": "data_seq/group2/", "startFrame": 303, "endFrame": 591, "nz": 6, "ext": "jpg", "anno_path": "anno/group2_2.txt"},
                {"name": "group2_3", "path": "data_seq/group2/", "startFrame": 591, "endFrame": 895, "nz": 6, "ext": "jpg", "anno_path": "anno/group2_3.txt"},
                {"name": "group3_1", "path": "data_seq/group3/", "startFrame": 1, "endFrame": 523, "nz": 6, "ext": "jpg", "anno_path": "anno/group3_1.txt"},
                {"name": "group3_2", "path": "data_seq/group3/", "startFrame": 523, "endFrame": 943, "nz": 6, "ext": "jpg", "anno_path": "anno/group3_2.txt"},
                {"name": "group3_3", "path": "data_seq/group3/", "startFrame": 943, "endFrame": 1457, "nz": 6, "ext": "jpg", "anno_path": "anno/group3_3.txt"},
                {"name": "group3_4", "path": "data_seq/group3/", "startFrame": 1457, "endFrame": 1843, "nz": 6, "ext": "jpg", "anno_path": "anno/group3_4.txt"},
                {"name": "person1", "path": "data_seq/person1/", "startFrame": 1, "endFrame": 267, "nz": 6, "ext": "jpg", "anno_path": "anno/person1.txt"},
                {"name": "person2_1", "path": "data_seq/person2/", "startFrame": 1, "endFrame": 397, "nz": 6, "ext": "jpg", "anno_path": "anno/person2_1.txt"},
                {"name": "person2_2", "path": "data_seq/person2/", "startFrame": 397, "endFrame": 875, "nz": 6, "ext": "jpg", "anno_path": "anno/person2_2.txt"},
                {"name": "person3", "path": "data_seq/person3/", "startFrame": 1, "endFrame": 215, "nz": 6, "ext": "jpg", "anno_path": "anno/person3.txt"},
                {"name": "person4_1", "path": "data_seq/person4/", "startFrame": 1, "endFrame": 501, "nz": 6, "ext": "jpg", "anno_path": "anno/person4_1.txt"},
                {"name": "person4_2", "path": "data_seq/person4/", "startFrame": 501, "endFrame": 915, "nz": 6, "ext": "jpg", "anno_path": "anno/person4_2.txt"},
                {"name": "person5_1", "path": "data_seq/person5/", "startFrame": 1, "endFrame": 293, "nz": 6, "ext": "jpg", "anno_path": "anno/person5_1.txt"},
                {"name": "person5_2", "path": "data_seq/person5/", "startFrame": 293, "endFrame": 701, "nz": 6, "ext": "jpg", "anno_path": "anno/person5_2.txt"},
                {"name": "person6", "path": "data_seq/person6/", "startFrame": 1, "endFrame": 301, "nz": 6, "ext": "jpg", "anno_path": "anno/person6.txt"},
                {"name": "person7_1", "path": "data_seq/person7/", "startFrame": 1, "endFrame": 417, "nz": 6, "ext": "jpg", "anno_path": "anno/person7_1.txt"},
                {"name": "person7_2", "path": "data_seq/person7/", "startFrame": 417, "endFrame": 689, "nz": 6, "ext": "jpg", "anno_path": "anno/person7_2.txt"},
                {"name": "person8_1", "path": "data_seq/person8/", "startFrame": 1, "endFrame": 359, "nz": 6, "ext": "jpg", "anno_path": "anno/person8_1.txt"},
                {"name": "person8_2", "path": "data_seq/person8/", "startFrame": 359, "endFrame": 509, "nz": 6, "ext": "jpg", "anno_path": "anno/person8_2.txt"},
                {"name": "person9", "path": "data_seq/person9/", "startFrame": 1, "endFrame": 221, "nz": 6, "ext": "jpg", "anno_path": "anno/person9.txt"},
                {"name": "person10", "path": "data_seq/person10/", "startFrame": 1, "endFrame": 341, "nz": 6, "ext": "jpg", "anno_path": "anno/person10.txt"},
                {"name": "person11", "path": "data_seq/person11/", "startFrame": 1, "endFrame": 241, "nz": 6, "ext": "jpg", "anno_path": "anno/person11.txt"},
                {"name": "person12_1", "path": "data_seq/person12/", "startFrame": 1, "endFrame": 201, "nz": 6, "ext": "jpg", "anno_path": "anno/person12_1.txt"},
                {"name": "person12_2", "path": "data_seq/person12/", "startFrame": 201, "endFrame": 541, "nz": 6, "ext": "jpg", "anno_path": "anno/person12_2.txt"},
                {"name": "person13", "path": "data_seq/person13/", "startFrame": 1, "endFrame": 295, "nz": 6, "ext": "jpg", "anno_path": "anno/person13.txt"},
                {"name": "person14_1", "path": "data_seq/person14/", "startFrame": 1, "endFrame": 283, "nz": 6, "ext": "jpg", "anno_path": "anno/person14_1.txt"},
                {"name": "person14_2", "path": "data_seq/person14/", "startFrame": 283, "endFrame": 605, "nz": 6, "ext": "jpg", "anno_path": "anno/person14_2.txt"},
                {"name": "person14_3", "path": "data_seq/person14/", "startFrame": 605, "endFrame": 975, "nz": 6, "ext": "jpg", "anno_path": "anno/person14_3.txt"},
                {"name": "person15", "path": "data_seq/person15/", "startFrame": 1, "endFrame": 447, "nz": 6, "ext": "jpg", "anno_path": "anno/person15.txt"},
                {"name": "person16", "path": "data_seq/person16/", "startFrame": 1, "endFrame": 383, "nz": 6, "ext": "jpg", "anno_path": "anno/person16.txt"},
                {"name": "person17_1", "path": "data_seq/person17/", "startFrame": 1, "endFrame": 501, "nz": 6, "ext": "jpg", "anno_path": "anno/person17_1.txt"},
                {"name": "person17_2", "path": "data_seq/person17/", "startFrame": 501, "endFrame": 783, "nz": 6, "ext": "jpg", "anno_path": "anno/person17_2.txt"},
                {"name": "person18", "path": "data_seq/person18/", "startFrame": 1, "endFrame": 465, "nz": 6, "ext": "jpg", "anno_path": "anno/person18.txt"},
                {"name": "person19_1", "path": "data_seq/person19/", "startFrame": 1, "endFrame": 415, "nz": 6, "ext": "jpg", "anno_path": "anno/person19_1.txt"},
                {"name": "person19_2", "path": "data_seq/person19/", "startFrame": 415, "endFrame": 931, "nz": 6, "ext": "jpg", "anno_path": "anno/person19_2.txt"},
                {"name": "person19_3", "path": "data_seq/person19/", "startFrame": 931, "endFrame": 1453, "nz": 6, "ext": "jpg", "anno_path": "anno/person19_3.txt"},
                {"name": "person20", "path": "data_seq/person20/", "startFrame": 1, "endFrame": 595, "nz": 6, "ext": "jpg", "anno_path": "anno/person20.txt"},
                {"name": "person21", "path": "data_seq/person21/", "startFrame": 1, "endFrame": 163, "nz": 6, "ext": "jpg", "anno_path": "anno/person21.txt"},
                {"name": "person22", "path": "data_seq/person22/", "startFrame": 1, "endFrame": 67, "nz": 6, "ext": "jpg", "anno_path": "anno/person22.txt"},
                {"name": "person23", "path": "data_seq/person23/", "startFrame": 1, "endFrame": 133, "nz": 6, "ext": "jpg", "anno_path": "anno/person23.txt"},
                {"name": "truck1", "path": "data_seq/truck1/", "startFrame": 1, "endFrame": 155, "nz": 6, "ext": "jpg", "anno_path": "anno/truck1.txt"},
                {"name": "truck2", "path": "data_seq/truck2/", "startFrame": 1, "endFrame": 129, "nz": 6, "ext": "jpg", "anno_path": "anno/truck2.txt"},
                {"name": "truck3", "path": "data_seq/truck3/", "startFrame": 1, "endFrame": 179, "nz": 6, "ext": "jpg", "anno_path": "anno/truck3.txt"},
                {"name": "truck4_1", "path": "data_seq/truck4/", "startFrame": 1, "endFrame": 193, "nz": 6, "ext": "jpg", "anno_path": "anno/truck4_1.txt"},
                {"name": "truck4_2", "path": "data_seq/truck4/", "startFrame": 193, "endFrame": 421, "nz": 6, "ext": "jpg", "anno_path": "anno/truck4_2.txt"},
                {"name": "uav1_1", "path": "data_seq/uav1/", "startFrame": 1, "endFrame": 519, "nz": 6, "ext": "jpg", "anno_path": "anno/uav1_1.txt"},
                {"name": "uav1_2", "path": "data_seq/uav1/", "startFrame": 519, "endFrame": 793, "nz": 6, "ext": "jpg", "anno_path": "anno/uav1_2.txt"},
                {"name": "uav1_3", "path": "data_seq/uav1/", "startFrame": 825, "endFrame": 1157, "nz": 6, "ext": "jpg", "anno_path": "anno/uav1_3.txt"},
                {"name": "uav2", "path": "data_seq/uav2/", "startFrame": 1, "endFrame": 45, "nz": 6, "ext": "jpg", "anno_path": "anno/uav2.txt"},
                {"name": "uav3", "path": "data_seq/uav3/", "startFrame": 1, "endFrame": 89, "nz": 6, "ext": "jpg", "anno_path": "anno/uav3.txt"},
                {"name": "uav4", "path": "data_seq/uav4/", "startFrame": 1, "endFrame": 53, "nz": 6, "ext": "jpg", "anno_path": "anno/uav4.txt"},
                {"name": "uav5", "path": "data_seq/uav5/", "startFrame": 1, "endFrame": 47, "nz": 6, "ext": "jpg", "anno_path": "anno/uav5.txt"},
                {"name": "uav6", "path": "data_seq/uav6/", "startFrame": 1, "endFrame": 37, "nz": 6, "ext": "jpg", "anno_path": "anno/uav6.txt"},
                {"name": "uav7", "path": "data_seq/uav7/", "startFrame": 1, "endFrame": 125, "nz": 6, "ext": "jpg", "anno_path": "anno/uav7.txt"},
                {"name": "uav8", "path": "data_seq/uav8/", "startFrame": 1, "endFrame": 101, "nz": 6, "ext": "jpg", "anno_path": "anno/uav8.txt"},
                {"name": "wakeboard1", "path": "data_seq/wakeboard1/", "startFrame": 1, "endFrame": 141, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard1.txt"},
                {"name": "wakeboard2", "path": "data_seq/wakeboard2/", "startFrame": 1, "endFrame": 245, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard2.txt"},
                {"name": "wakeboard3", "path": "data_seq/wakeboard3/", "startFrame": 1, "endFrame": 275, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard3.txt"},
                {"name": "wakeboard4", "path": "data_seq/wakeboard4/", "startFrame": 1, "endFrame": 233, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard4.txt"},
                {"name": "wakeboard5", "path": "data_seq/wakeboard5/", "startFrame": 1, "endFrame": 559, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard5.txt"},
                {"name": "wakeboard6", "path": "data_seq/wakeboard6/", "startFrame": 1, "endFrame": 389, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard6.txt"},
                {"name": "wakeboard7", "path": "data_seq/wakeboard7/", "startFrame": 1, "endFrame": 67, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard7.txt"},
                {"name": "wakeboard8", "path": "data_seq/wakeboard8/", "startFrame": 1, "endFrame": 515, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard8.txt"},
                {"name": "wakeboard9", "path": "data_seq/wakeboard9/", "startFrame": 1, "endFrame": 119, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard9.txt"},
                {"name": "wakeboard10", "path": "data_seq/wakeboard10/", "startFrame": 1, "endFrame": 157, "nz": 6, "ext": "jpg", "anno_path": "anno/wakeboard10.txt"},
                {"name": "car1_s", "path": "data_seq/car1_s/", "startFrame": 1, "endFrame": 492, "nz": 6, "ext": "jpg", "anno_path": "anno/car1_s.txt"},
                {"name": "car2_s", "path": "data_seq/car2_s/", "startFrame": 1, "endFrame": 107, "nz": 6, "ext": "jpg", "anno_path": "anno/car2_s.txt"},
                {"name": "car3_s", "path": "data_seq/car3_s/", "startFrame": 1, "endFrame": 434, "nz": 6, "ext": "jpg", "anno_path": "anno/car3_s.txt"},
                {"name": "car4_s", "path": "data_seq/car4_s/", "startFrame": 1, "endFrame": 277, "nz": 6, "ext": "jpg", "anno_path": "anno/car4_s.txt"},
                {"name": "person1_s", "path": "data_seq/person1_s/", "startFrame": 1, "endFrame": 534, "nz": 6, "ext": "jpg", "anno_path": "anno/person1_s.txt"},
                {"name": "person2_s", "path": "data_seq/person2_s/", "startFrame": 1, "endFrame": 84, "nz": 6, "ext": "jpg", "anno_path": "anno/person2_s.txt"},
                {"name": "person3_s", "path": "data_seq/person3_s/", "startFrame": 1, "endFrame": 169, "nz": 6, "ext": "jpg", "anno_path": "anno/person3_s.txt"},
            ]
        return sequence_info_list
