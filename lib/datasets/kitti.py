# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# This file is tested only for end2end with RPN mode.
# --------------------------------------------------------
# If you add another dataset,
#   please modify follow files.
#       - json instances (converted raw annotation file)
#       - this file
#       - roi_data_layer/minibatch.py (input layer)
#       - rpn/anchor_target_layer.py (generate GT for RPN)
#       - rpn/proposal_layer.py (produce RoIs in pixel: sort, nms)
#       - rpn/proposal_target_layer.py (generate GT for RCNN)
# --------------------------------------------------------
#
# For KITTI, 
#       Ignore instances: do not load (filtered out at func. self._KITTI.getAnnIds)
#       Empty images: add dummy gt_box
#           gt_classes[0] = 0
#           overlaps[0, :] = -1.0
#
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid
# COCO API
from pycocotools.kitti import KITTI
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

class kitti(imdb):

    def get_anchors(self):
        # Data-driven anchors which are defined by K-means clustering        
        # anchors = np.array( [[ -58.083,  -33.423,   58.083,   33.423],  \
        #                     [ -15.409,  -11.599,   15.409,   11.599],   \
        #                     [ -29.462,  -19.389,   29.462,   19.389],   \
        #                     [  -9.935,  -25.517,    9.935,   25.517],   \
        #                     [ -29.315,  -66.092,   29.315,   66.092],   \
        #                     [-117.351,  -69.798,  117.351,   69.798]])

        anchors = np.array([   [  -7.94,  -22.93,    7.94,   22.93], \
                               [ -16.93,  -15.07,   16.93,   15.07], \
                               [ -24.88,  -19.1 ,   24.88,   19.1 ], \
                               [ -39.43,  -17.01,   39.43,   17.01], \
                               [ -16.19,  -41.83,   16.19,   41.83], \
                               [ -34.88,  -28.32,   34.88,   28.32], \
                               [ -61.55,  -25.8 ,   61.55,   25.8 ], \
                               [ -56.78,  -42.03,   56.78,   42.03], \
                               [ -36.25,  -79.62,   36.25,   79.62], \
                               [ -98.59,  -44.44,   98.59,   44.44], \
                               [ -96.43,  -78.41,   96.43,   78.41], \
                               [-165.34,  -93.99,  165.34,   93.99]])

        return anchors

    def __init__(self, image_set, year):
        imdb.__init__(self, 'kitti_' + year + '_' + image_set)
        # KITTI specific config options
        self.config = {'cleanup' : True,                       
                       'hRng' : [25, np.inf], # Min. 20 x 50 or 25 x 40
                       'occLevel' : [0, 1, 2],       # 0: fully visible, 1: partly occ, 2: largely occ, 3: unknown
                       'truncRng' : [0, 0.5]     # Only partially-truncated
                      }

        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'kitti')
        
        # load KITTI API, classes, class <-> id mappings
        self._KITTI = KITTI(self._get_ann_file())

        # Below classes are only used for training.
        categories = ['Pedestrian', 'Cyclist', 'Car', 'Person_sitting', 'Van', 'Truck', 'Tram', 'Misc']
        self._raw_cat_ids = self._KITTI.getCatIds(catNms=categories)

        cats = self._KITTI.loadCats(self._raw_cat_ids)
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_kitti_cat_id = dict(zip([c['name'] for c in cats], self._raw_cat_ids))

        self._image_index = self._load_image_set_index()
        
        # Default to roidb handler        
        self.set_proposal_method('gt')
        #self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.

        # For KITTI dataset, raw-train set provided by the original author is divided into train/val set.
        #   So, we call raw-train set trainval2012 consisting of train2012 and val2012.        
        self._view_map = {            
            'val2012' : 'trainval2012',
            'train2012' : 'trainval2012'
        }
        
        # E.g. train2012/val2012 -> self._data_name = 'trainval2012'
        #      test2012 -> self._data_name = 'test2012'
        kitti_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[kitti_name]
                           if self._view_map.has_key(kitti_name)
                           else kitti_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        #self._gt_splits = ['train', 'val', 'minival']

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
                             else 'image_info'            
        
        return osp.join(self._data_path, 'annotations',
                        prefix + '_' + self._image_set + self._year + '.json')        

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._KITTI.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._KITTI.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        im_ann = self._KITTI.loadImgs(index)[0]                    
        image_path = osp.join(self._data_path, 'images', self._data_name, im_ann['file_name'])
        
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # def _roidb_from_proposals(self, method):
    #     """
    #     Creates a roidb from pre-computed proposals of a particular methods.
    #     """
    #     top_k = self.config['top_k']
    #     cache_file = osp.join(self.cache_path, self.name +
    #                           '_{:s}_top{:d}'.format(method, top_k) +
    #                           '_roidb.pkl')

    #     if osp.exists(cache_file):
    #         with open(cache_file, 'rb') as fid:
    #             roidb = cPickle.load(fid)
    #         print '{:s} {:s} roidb loaded from {:s}'.format(self.name, method,
    #                                                         cache_file)
    #         return roidb

    #     if self._image_set in self._gt_splits:
    #         gt_roidb = self.gt_roidb()
    #         method_roidb = self._load_proposals(method, gt_roidb)
    #         roidb = imdb.merge_roidbs(gt_roidb, method_roidb)
    #         # Make sure we don't use proposals that are contained in crowds
    #         #roidb = _filter_crowd_proposals(roidb, self.config['crowd_thresh'])
    #     else:
    #         roidb = self._load_proposals(method, None)
    #     with open(cache_file, 'wb') as fid:
    #         cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    #     print 'wrote {:s} roidb to {:s}'.format(method, cache_file)
    #     return roidb

    # def _load_proposals(self, method, gt_roidb):
    #     """
    #     Load pre-computed proposals in the format provided by Jan Hosang:
    #     http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
    #       computing/research/object-recognition-and-scene-understanding/how-
    #       good-are-detection-proposals-really/
    #     For MCG, use boxes from http://www.eecs.berkeley.edu/Research/Projects/
    #       CS/vision/grouping/mcg/ and convert the file layout using
    #     lib/datasets/tools/mcg_munge.py.
    #     """
    #     box_list = []
    #     top_k = self.config['top_k']
    #     valid_methods = [
    #         'MCG',
    #         'selective_search',
    #         'edge_boxes_AR',
    #         'edge_boxes_70']
    #     assert method in valid_methods

    #     print 'Loading {} boxes'.format(method)
    #     for i, index in enumerate(self._image_index):
    #         if i % 1000 == 0:
    #             print '{:d} / {:d}'.format(i + 1, len(self._image_index))

    #         box_file = osp.join(
    #             cfg.DATA_DIR, 'kitti_proposals', method, 'mat',
    #             self._get_box_file(index))

    #         raw_data = sio.loadmat(box_file)['boxes']
    #         boxes = np.maximum(raw_data - 1, 0).astype(np.uint16)
    #         if method == 'MCG':
    #             # Boxes from the MCG website are in (y1, x1, y2, x2) order
    #             boxes = boxes[:, (1, 0, 3, 2)]
    #         # Remove duplicate boxes and very small boxes and then take top k
    #         keep = ds_utils.unique_boxes(boxes)
    #         boxes = boxes[keep, :]
    #         keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
    #         boxes = boxes[keep, :]
    #         boxes = boxes[:top_k, :]
    #         box_list.append(boxes)
    #         # Sanity check
    #         im_ann = self._COCO.loadImgs(index)[0]
    #         width = im_ann['width']
    #         height = im_ann['height']
    #         ds_utils.validate_boxes(boxes, width=width, height=height)
    #     return self.create_roidb_from_box_list(box_list, gt_roidb)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_kitti_annotation(self, index):
        """
        Loads KITTI bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._KITTI.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        # Follow 'demo_load_kitti_dataset.py by Soonmin'        
        hRng, occLevel, tRng = self.config['hRng'], self.config['occLevel'], self.config['truncRng']

        # Load annotation ids
        annIds = self._KITTI.getAnnIds(imgIds=index, catIds=self._raw_cat_ids, 
                                       hRng=hRng, occLevel=occLevel, truncRng=tRng)
        #annIds = self._KITTI.getAnnIds(imgIds=index, hRng=hRng, occLevel=occLevel, truncRng=tRng)
        
        objs = self._KITTI.loadAnns(annIds)        

        # Sanitize bboxes -- some are invalid

        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))            
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            
            # All valid annotations must satisfy below condition
            if obj['area'] >= 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        objs = valid_objs            
        num_objs = len(objs)

        if num_objs == 0:
            # In traffic scene datasets (e.g. KITTI, KAIST),
            #   some images may not contain any target object instance.            
            
            # Fill dummy gt_boxes with [x1, y1, x2, y2] = [1, 1, 2, 2]
            boxes = np.zeros((1, 4), dtype=np.uint16)
            gt_classes = np.zeros((1), dtype=np.int32)
            overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
            seg_areas = np.zeros((1), dtype=np.float32)

            boxes[0, :] = [1, 1, 2, 2]
            gt_classes[0] = 0
            overlaps[0, :] = -1.0
            seg_areas[0] = 1
                    
            overlaps = scipy.sparse.csr_matrix(overlaps) 
            
            return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}
                #'seg_areas' : seg_areas}

        else:            
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            # Lookup table to map from KITTI category ids to our internal class indices                        
            kitti_cat_id_to_class_ind = dict([(self._class_to_kitti_cat_id[cls], self._class_to_ind[cls])
                                             for cls in self._classes[1:]])
                        
            for ix, obj in enumerate(objs):
                cls = kitti_cat_id_to_class_ind[ obj['category_id'] ]
                boxes[ix, :] = obj['clean_bbox']
                gt_classes[ix] = cls                
                overlaps[ix, cls] = 1.0
                                
            ds_utils.validate_boxes(boxes, width=width, height=height)
            overlaps = scipy.sparse.csr_matrix(overlaps)        

            return {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False}
                    #'seg_areas' : seg_areas}

    # def _get_box_file(self, index):
    #     # first 14 chars / first 22 chars / all chars + .mat
    #     # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        
    #     assert false, 'Do not use this function'

    #     file_name = ('COCO_' + self._data_name +
    #                  '_' + str(index).zfill(12) + '.mat')
    #     return osp.join(file_name[:14], file_name[:22], file_name)

    # def _print_detection_eval_metrics(self, kitti_eval):
    #     IoU_lo_thresh = 0.5
    #     IoU_hi_thresh = 0.95
    #     def _get_thr_ind(kitti_eval, thr):
    #         ind = np.where((kitti_eval.params.iouThrs > thr - 1e-5) &
    #                        (kitti_eval.params.iouThrs < thr + 1e-5))[0][0]
    #         iou_thr = kitti_eval.params.iouThrs[ind]
    #         assert np.isclose(iou_thr, thr)
    #         return ind

    #     ind_lo = _get_thr_ind(kitti_eval, IoU_lo_thresh)
    #     ind_hi = _get_thr_ind(kitti_eval, IoU_hi_thresh)
    #     # precision has dims (iou, recall, cls, area range, max dets)
    #     # area range index 0: all area ranges
    #     # max dets index 2: 100 per image
    #     precision = \
    #         kitti_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    #     ap_default = np.mean(precision[precision > -1])
    #     print ('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
    #            '~~~~').format(IoU_lo_thresh, IoU_hi_thresh)
    #     print '{:.1f}'.format(100 * ap_default)
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         # minus 1 because of __background__
    #         precision = kitti_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
    #         ap = np.mean(precision[precision > -1])
    #         print '{:.1f}'.format(100 * ap)

    #     print '~~~~ Summary metrics ~~~~'
    #     kitti_eval.summarize()

    # def _do_detection_eval(self, res_file, output_dir):
    #     ann_type = 'bbox'
    #     kitti_dt = self._KITTI.loadRes(res_file)
    #     kitti_eval = COCOeval(self._KITTI, kitti_dt)
    #     kitti_eval.params.useSegm = (ann_type == 'segm')
    #     kitti_eval.evaluate()
    #     kitti_eval.accumulate()
    #     self._print_detection_eval_metrics(kitti_eval)
    #     eval_file = osp.join(output_dir, 'detection_results.pkl')
    #     with open(eval_file, 'wb') as fid:
    #         cPickle.dump(kitti_eval, fid, cPickle.HIGHEST_PROTOCOL)
    #     print 'Wrote KITTI eval results to: {}'.format(eval_file)

    def _kitti_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in xrange(dets.shape[0])])
        return results

    def _write_kitti_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                          self.num_classes - 1)
            kitti_cat_id = self._class_to_kitti_cat_id[cls]
            results.extend(self._kitti_results_one_category(all_boxes[cls_ind],
                                                           kitti_cat_id))            
        print 'Writing results json to {}'.format(res_file)
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
      

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         self._year +
                                         '_results'))
        #if self.config['use_salt']:
        #    res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_kitti_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        #if self._image_set.find('test') == -1:
        #    self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        #if self.config['cleanup']:
        #    os.remove(res_file)

    #def competition_mode(self, on):
    #    if on:
    #        self.config['use_salt'] = False
    #        self.config['cleanup'] = False
    #    else:
    #        self.config['use_salt'] = True
    #        self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.kitti import kitti
    d = kitti('train', '2012')
    res = d.roidb
    from IPython import embed; embed()

