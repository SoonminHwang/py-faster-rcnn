#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Modified by Soonmin Hwang
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
Modified to enhance flexivility for trained model selection and dataset

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

VOC_CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
KITTI_CLASSES = ('__background__', 'Pedestrian', 'Cyclist', 'Car')

DATASETS = {'kitti': KITTI_CLASSES, 'voc': VOC_CLASSES}

#CONF_THRESH = 0.8
CONF_THRESH = 0.2
NMS_THRESH = 0.3

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()    
    plt.show()


def demo(net, image_name, conf_thres, nms_thres, resDir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    #fp = open( os.path.join(cfg.DATA_DIR, 'demo', image_name.split('.')[0]+'.txt'), 'w' )
    fp = open( os.path.join(resDir, os.path.basename(image_name).split('.')[0]+'.txt'), 'w' )

    results = np.zeros((0, 6), dtype=np.float32)

    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thres)
        dets = dets[keep, :]

	#import pdb
	#pdb.set_trace()

	results = np.vstack( (results, np.insert(dets, 0, cls_ind, axis=1)) )
	#results.append({'cls':cls,'dets':dets})
	#res = np.hstack((np.repeat(np.asarray(cls_ind), len(dets), axis=0), dets))
	#results = np.vstack((results, ))
	#for d in dets:
  	#    fp.write( '{:s} {:.2f} {:.2f} {:.2f} {:.2f} {:.4f}\n'.format(cls, d[0], d[1], d[2], d[3], d[4]) )

        #vis_detections(im, cls, dets, thresh=conf_thres)
        #plt.savefig(os.path.join(cfg.DATA_DIR, 'demo', '[Result]' + image_name))

    print('# of results: {}'.format(len(results)))

    #import pdb
    #pdb.set_trace()
    idx = np.argsort(results[:,-1])
    results = results[idx[::-1],:]
    #results = np.sort(results, axis=-1)[-1:0:-1, :]
    #results = results[-1:0:-1, :]

    for res in results:
	resStr = '{:s} '.format(CLASSES[int(res[0])])
	resStr += '-1 -1 -10 ' # Default values for truncation, occlusion, alpha
	resStr += ' {:.2f} {:.2f} {:.2f} {:.2f} '.format(
		res[1],res[2],res[3],res[4])	# x1 y1 x2 y2
	resStr += '-1 -1 -1 -1000 -1000 -1000 -10 {:.2f}\n'.format(res[5])
	fp.write( resStr )

    fp.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [*.prototxt]')
    parser.add_argument('--caffemodel', dest='caffemodel', help='Trained weights [*.caffemodel]')
    parser.add_argument('--dataset', dest='dataset', help='Specify the trained dataset for category definition',
                        choices=DATASETS.keys(), default='kitti')

    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)
    parser.add_argument('--method', dest='method_name', help='Algorithm name', type=str)
    parser.add_argument('--evalDir', dest='eval_dir', 
			help='For evaluation, please specify evaluation directory',
			default='', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])

    CLASSES = DATASETS[args.dataset]

    prototxt = args.demo_net
    caffemodel = args.caffemodel
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #            'KITTI_003313.png', 'KITTI_000023.png', 'KITTI_000211.png', 'KITTI_001443.png' ] # KITTI Test set

    import datetime
    nowStr = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    resDir = os.path.join(cfg.DATA_DIR, 'kitti', 'evaluation', 'results', nowStr + '_' + args.method_name)
    if not os.path.exists(resDir):
        os.makedirs( resDir )

    if args.eval_dir is not '':
	    import glob
	    im_names = glob.glob( os.path.join(args.eval_dir, '*.png') )
	    print( 'Get test images from {}'.format(args.eval_dir) )

	    for im_name in im_names:
		demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
    else:

        im_names = ['KITTI_000017.png']
        #im_names = ['KITTI_003313.png', 'KITTI_000023.png', 'KITTI_000211.png', 'KITTI_001443.png',
        #		'KITTI_000017.png', 'KITTI_000031.png', 'KITTI_000040.png' ] # KITTI Test set]
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
	    im_name = os.path.join( cfg.DATA_DIR, 'demo', im_name )
            demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
     
    #plt.show()
