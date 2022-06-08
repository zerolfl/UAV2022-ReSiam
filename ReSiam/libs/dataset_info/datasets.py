from collections import namedtuple
import importlib

from .data import SequenceList


DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])  # 具名元组，元组有名称，其中的元素也有名称

pt = "libs.dataset_info.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    uav123=DatasetInfo(module=pt % "uav123", class_name="UAV123Dataset", kwargs=dict(version='30fps')),
    uav20l=DatasetInfo(module=pt % "uav123", class_name="UAV123Dataset", kwargs=dict(version='20l')),
    uav123_10fps=DatasetInfo(module=pt % "uav123", class_name="UAV123Dataset", kwargs=dict(version='10fps')),
    dtb70=DatasetInfo(module=pt % "dtb70", class_name="DTB70Dataset", kwargs=dict()),
    uav2022resiam=DatasetInfo(module=pt % "uav2022resiam", class_name="UAV2022ReSiamDataset", kwargs=dict()),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
