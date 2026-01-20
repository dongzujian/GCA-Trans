"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .transparent11 import TransparentSegmentation
from .gdd import gdd
from .gsd import gsd
from .rgbdgsd import rgbdgsd
datasets = {
    'transparent11': TransparentSegmentation, 
    'gdd': gdd,
    'gsd': gsd,
    'rgbdgsd':rgbdgsd
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
