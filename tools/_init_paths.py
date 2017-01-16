# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
#caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
caffe_path = osp.join(this_dir, '..', 'caffe-latest', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)


# python startup file 
import readline 
import rlcompleter 
import atexit 
import os 
# tab completion 
readline.parse_and_bind('tab: complete') 
# history file 
histfile = os.path.join(os.environ['HOME'], '.pythonhistory') 
try: 
    readline.read_history_file(histfile) 
except IOError: 
    pass 
atexit.register(readline.write_history_file, histfile) 
del os, histfile, readline, rlcompleter