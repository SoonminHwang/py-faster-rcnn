# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.kaist_ped import kaist_ped
from datasets.caltech_ped import caltech_ped

import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up kaist_ped_2015_<split>
for year in ['2015']:	
	for split in ['train20', 'train02', 'test20', 'test01']:	# test01: to generate result videos
        name = 'kaist_ped_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: kaist_ped(split, year))
		
# Set up caltech_ped_2009_<split>
for year in ['2009']:	
	for split in ['train30', 'train04', 'test30', 'test01']:	# test01: to generate result videos
        name = 'caltech_ped_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: caltech_ped(split, year))

		
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
