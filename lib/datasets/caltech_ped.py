# --------------------------------------------------------
# Caltech Pedestrian Dataset Loader
# 	Written by Soonmin Hwang (RCV Lab., KAIST)
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
#import xml.etree.ElementTree as ET
import json
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import re
#from voc_eval import voc_eval
from fast_rcnn.config import cfg

class caltech_ped(imdb):
    def __init__(self, image_set, year, config=None):
        imdb.__init__(self, 'caltech_ped_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        
        self._data_path = self._get_default_path()
        self._classes = ('__background__', # always index 0
                         'person', 'ignore')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        		
	# Default to roidb handler
        self._roidb_handler = self.rpn_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp1'		# usage?

        # Caltech specific config options: Reasonable (default) for training
	# pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}, ...
	#		'hRng',[50 inf], 'vRng',[1 1]};		
        self.config = {'labels'      : ['person'],
                       'ign_labels'  : ['people'],
                       'squarify'    : [3, 0.41],                       
                       'vRng'        : [1, 1],
		       'xRng'        : [0, 640],
		       'yRng'        : [0, 480],
		       'wRng' 	     : [-np.inf, np.inf],
		       'hRng'        : [50, np.inf],
                       'use_flip'    : True
		       'matlab_eval' : False}
		
		# To do: write config-merge code
        
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:	
        # self._data_path + /ImageSets/ + self._image_set + .txt (e.g. train04, test30)
		# 
		# In image set file,
		#		set00/set00_V001_I0000029
		#		set00/set00_V001_I0000059
		#		set00/set00_V001_I0000089
        image_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where Caltech Pedestrian Dataset is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'caltech')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

	# Load annotation file
	#filename = os.path.join(self._data_path, 'annotations.json')
	#with open(filename, 'r') as f:
	#	gt = json.loads(f.read())[0]
				
        gt_roidb = [self._load_caltech_annotation(index)
                    for index in self.image_index]		
		
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test30' or self._image_set != 'test01':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self):
        if self._image_set != 'test30' or self._image_set != 'test01':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    

    def _load_caltech_annotation(self, index):
        """
        Load image and bounding boxes info from .txt file in the Caltech format.
        """
        filename = os.path.join(self.DATA_DIR, 'annotations', index + '.txt')
	# print 'Loading: {}'.format(filename)
	with open(filename) as f:
            data = f.read()
	import re
	# Caltech-stype annotation format (class, x, y, w, h, occ_x, occ_y, occ_w, occ_h, ...)
	# e.g. Person 10 10 
	objs = re.findall('(\S+,) (\d*\.?\d*) \2 \2 \2 \2 \2 \2 \2', data)
	
	num_objs = len(objs)
	
	boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.uint8)
        
	# Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
	    ignore = false
	
            # Make pixel indexes 0-based
	    coor = re.findall('(\d*\.?\d*)', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2]) + x1
            y2 = float(coor[3]) + y1
	
            
            vx1 = float(coor[4])
            vy1 = float(coor[5])
            vx2 = float(coor[6]) + vx1
            vy2 = float(coor[7]) + vy1
		
	    # x range
            if x1 < self.config['xRng'][0] or x1 > self.config['xRng'][1]: ignore = True
	    if x2 < self.config['xRng'][0] or x2 > self.config['xRng'][1]: ignore = True            
				
            # y range
            if y1 < self.config['yRng'][0] or y1 > self.config['yRng'][1]: ignore = True
            if y2 < self.config['yRng'][0] or y2 > self.config['yRng'][1]: ignore = True			
            
            # w range
            if x2 - x1 < self.config['wRng'][0] or x2 - x1 > self.config['wRng'][1]: ignore = True				
            # h range
            if y2 - y1 < self.config['hRng'][0] or y2 - y1 > self.config['hRng'][1]: ignore = True
				
            # v range
            if (x,y,w,h) == (vx,vy,vw,vh):
                v = 0
            elif (vx,vy,vw,vh) == 0:
                v = 1 
            else:
                v = (vw*vh)/(w*h)
  
            if v < self.config['vRng'][0] or v > self.config['vRng'][1]: ignore = True
	
	    # Ignore label,
            if obj[0] not in self.config['labels']:	ignore = True		
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = self._class_to_ind['person'] if not ignore else self._class_to_ind['ignore']
	
	
	
	#set_name, video_name, n_frame = os.path.basename(index).split('_')	
	#n_frame = re.search('([0-9]+)\' + self._image_ext, index).groups()[0]
		
	#	if n_frame in annotations[set_name][video_name]['frames']:
	#		data = annotations[set_name][video_name]['frames'][n_frame]
	#		for datum in data:
	#			
	#			ignore = False
	#			
	#			if self.config['flipped']:
	#				is_flip = random.choice(2) * 2 - 1
	#			else:
	#				is_filp = 1
	#			
	#			# filter objs (set ignore flags)
	#			x, y, w, h = [int(v) for v in datum['pos']]					
	#			vx, vy, vw, vh = [int(v) for v in datum['posv']]
	#			
	#			if is_flip:
	#				x, y = [cfg.IMAGE_WIDTH - x - w, cfg.IMAGE_HEIGHT - y - h]
	#				vx, vy = [cfg.IMAGE_WIDTH - vx - vw, cfg.IMAGE_HEIGHT - vy - vh]
	#				
	#			lbl = datum['lbl']
	#			
	#			# Will not be loaded
	#			if lbl not in self.config['labels']:	continue
	#			
	#			# x range
	#			if x < self.config['xRng'][0] or x > self.config['xRng'][1]: ignore = True
	#			if x + w < self.config['xRng'][0] or x + w > self.config['xRng'][1]: ignore = True
	#			
	#			# y range
	#			if y < self.config['yRng'][0] or y > self.config['yRng'][1]: ignore = True
	#			if y + h < self.config['yRng'][0] or y + h > self.config['yRng'][1]: ignore = True
	#			
	#			# w range
	#			if w < self.config['wRng'][0] or w > self.config['wRng'][1]: ignore = True
	#			
	#			# h range
	#			if h < self.config['hRng'][0] or h > self.config['hRng'][1]: ignore = True
	#			
	#			# v range
	#			if (x,y,w,h) == (vx,vy,vw,vh):
	#				v = 0
	#			elif (vx,vy,vw,vh) == 0:
	#				v = 1 
	#			else:
	#				v = (vw*vh)/(w*h)
	#
	#			if v < self.config['vRng'][0] or v > self.config['vRng'][1]: ignore = True
	#			
	#			# Ignore label,
	#			if lbl not in self.config['labels']:	ignore = True
	#			
	#			boxes = np.vstack((boxes, [x, y, w, h]))
	#			classes = np.vstack((classes, self._class_to_ind['ignore' if ignore else 'pedestrian']))
	#			flipped = np.vstack((flipped, is_flip))
	#	else:
	#		print("Error!! index image is not in annotations.json")
	#		
		return {'boxes' : boxes,
                'gt_classes': classes,                
                'flipped' : false}
		
    #def _get_comp_id(self):
    #    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
    #        else self._comp_id)
    #    return comp_id

    #def _get_voc_results_file_template(self):
    #    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #    path = os.path.join(
    #        self._devkit_path,
    #        'results',
    #        'VOC' + self._year,
    #        'Main',
    #        filename)
    #    return path

    #def _write_voc_results_file(self, all_boxes):
    #    for cls_ind, cls in enumerate(self.classes):
    #        if cls == '__background__':
    #            continue
    #        print 'Writing {} VOC results file'.format(cls)
    #        filename = self._get_voc_results_file_template().format(cls)
    #        with open(filename, 'wt') as f:
    #            for im_ind, index in enumerate(self.image_index):
    #                dets = all_boxes[cls_ind][im_ind]
    #                if dets == []:
    #                    continue
    #                # the VOCdevkit expects 1-based indices
    #                for k in xrange(dets.shape[0]):
    #                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                            format(index, dets[k, -1],
    #                                   dets[k, 0] + 1, dets[k, 1] + 1,
    #                                   dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        pass

    def _do_matlab_eval(self, output_dir = 'output'):
        pass

    def evaluate_detections(self, all_boxes, output_dir):        
	self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        
	if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
#        if self.config['cleanup']:
#            for cls in self._classes:
#                if cls == '__background__':
#                    continue
#                filename = self._get_voc_results_file_template().format(cls)
#                os.remove(filename)

#    def competition_mode(self, on):
#        if on:
#            self.config['use_salt'] = False
#            self.config['cleanup'] = False
#        else:
#            self.config['use_salt'] = True
#            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.caltech_ped import caltech_ped
    d = caltech_ped('train04', '2009')
    res = d.roidb
    from IPython import embed; embed()
