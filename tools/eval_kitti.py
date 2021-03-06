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
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.cython_bbox import bbox_overlaps

from calc_mAP_KITTI import calculate_mAP as calc_mAP

from datasets.factory import get_imdb

CLASSES = ('__background__', 'Pedestrian', 'Cyclist', 'Car')

#CONF_THRESH = 0.8
CONF_THRESH = 0.2
NMS_THRESH = 0.3

NETS = {'VGG16': ('VGG16',
                  'vgg16_faster_rcnn_iter_{:d}.caffemodel'),
        'ZF': ('ZF',
                  'zf_faster_rcnn_iter_{:d}.caffemodel'),
        'ResNet50': ('ResNet50',
                  'resnet50_faster_rcnn_iter_{:d}.caffemodel'),
        'AlexNet': ('AlexNet',
                  'alexnet_faster_rcnn_iter_{:d}.caffemodel')}

def demo(net, image_name, conf_thres, nms_thres, resDir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    fname = os.path.basename(image_name)

    # Detect all object classes and regress object bounds
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer(), 'save' : Timer()}

    _t['im_detect'].tic()
    #cfg_from_list(['TEST.SCALES', '[300]'])
    #scores, boxes = im_detect(net, im)

    cfg_from_list(['TEST.SCALES', '[576]'])
    scores, boxes = im_detect(net, im)

    #cfg_from_list(['TEST.SCALES', '[750]'])
    #scores3, boxes3 = im_detect(net, im)
    _t['im_detect'].toc()
    
    
    _t['misc'].tic()
    
    #scores = np.vstack( (scores, scores2, scores3) )
    #boxes = np.vstack( (boxes, boxes2, boxes3) )

    results = np.zeros((0, 6), dtype=np.float32)
    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):        
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        # CPU NMS is much faster than GPU NMS when the number of boxes
        # is relative small (e.g., < 10k)
        # TODO(rbg): autotune NMS dispatch
        keep = nms(dets, nms_thres, force_cpu=True)
        dets = dets[keep, :]
        results = np.vstack( (results, np.insert(dets, 0, cls_ind, axis=1)) )        
    _t['misc'].toc()  

    _t['save'].tic()
    with open( os.path.join(resDir, fname.split('.')[0] + '.txt'), 'w') as fp:        
        for det in results:
            if len(det) == 0: continue        
            if det[5] < 0.01: continue

            resStr = '{:s} -1 -1 -10 '.format(CLASSES[int(det[0])])                                
            resStr += ' {:.2f} {:.2f} {:.2f} {:.2f} '.format(det[1],det[2],det[3],det[4])    # x1 y1 x2 y2
            resStr += '-1 -1 -1 -1000 -1000 -1000 -10 {:.4f}\n'.format(det[5])
            fp.write( resStr )
            

    _t['save'].toc()
    return _t
       

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [resnet50]',
                        choices=NETS.keys(), default='resnet50')
    #parser.add_argument('--caffemodel', dest='caffemodel', help='Trained weights [*.caffemodel]')
    parser.add_argument('--iter', dest='demo_iter', help='Iteration', type=int)
    parser.add_argument('--exp', dest='demo_exp', help='Exp. name', type=str)


    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)    

    default_eval_dir = os.path.join(cfg.DATA_DIR, 'kitti', 'images', 'val2012')
    parser.add_argument('--evalDir', dest='eval_dir', 
			help='For evaluation, please specify evaluation directory',
			default=default_eval_dir, type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    cfg_from_file('./experiments/cfgs/faster_rcnn_end2end_kitti.yml')

    cfg_from_list(['TEST.SCALES', '[300]'])
    print('cfg.TEST.SCALES: {}'.format(cfg.TEST.SCALES)),

    cfg_from_list(['TEST.SCALES', '[750]'])
    print('cfg.TEST.SCALES: {}'.format(cfg.TEST.SCALES)),

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])

    prototxt = os.path.join(cfg.MODELS_DIR, 'kitti', NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    
    model = NETS[args.demo_net][1].format(args.demo_iter)
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', args.demo_exp, 
        'kitti_2012_train', model)
        
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # imdb
    #imdb = combined_roidb('kitti_2012_val')
    imdb = get_imdb('kitti_2012_val')

    if cfg.TRAIN.DATADRIVEN_ANCHORS:
        # Set dataset-specific anchors
        anchors = imdb.get_anchors()

        #anchor_target_layer_ind = list(net._layer_names).index('rpn-data')
        #net.layers[anchor_target_layer_ind].setup_anchor(anchors)

        proposal_layer_ind = list(net._layer_names).index('proposal')
        net.layers[proposal_layer_ind].setup_anchor(anchors)


    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    import datetime
    nowStr = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    #nowStr = '2016-12-28_11h02m'
    #resDir = os.path.join(cfg.DATA_DIR, 'kitti', 'evaluation', 'results', nowStr + '_' + args.method_name)
    #if not os.path.exists(resDir):
    #    os.makedirs( resDir )
    
    # Save detections    
    resDir = os.path.join(cfg.ROOT_DIR, 'output', args.demo_exp, 'kitti_2012_val', 
        nowStr, model.split('.')[0])
    if not os.path.exists(resDir):
        os.makedirs( resDir )

    if args.eval_dir is not '':
        import glob
        im_names = glob.glob( os.path.join(args.eval_dir, '*.png') )
        print( 'Get test images from {}'.format(args.eval_dir) )

        for ii, im_name in enumerate(im_names):
            timer = demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(ii + 1, 
                len(im_names), timer['im_detect'].average_time, 
                timer['misc'].average_time, timer['save'].average_time)

        try:

            os.system('cp tools/kitti_evaluate_object {:s}'.format(resDir))
            os.chdir(resDir)
            os.system('./kitti_evaluate_object')      
            
            print '---------------------------------------------'
            print '|  Class      |  Easy  | Moderate  |  Hard  |'
            print '---------------------------------------------'
            rec, mAP = calc_mAP('Car', 'plot')
            print '| Car         | {:.2f} |  {:.2f}   | {:.2f} |'.format(mAP['Easy'], mAP['Moderate'], mAP['Hard'])
            rec, mAP = calc_mAP('Pedestrian', 'plot')
            print '| Pedestrian  | {:.2f} |  {:.2f}   | {:.2f} |'.format(mAP['Easy'], mAP['Moderate'], mAP['Hard'])
            rec, mAP = calc_mAP('Cyclist', 'plot')
            print '| Cyclist     | {:.2f} |  {:.2f}   | {:.2f} |'.format(mAP['Easy'], mAP['Moderate'], mAP['Hard'])
            print '---------------------------------------------'
        
        except:
            import pdb
            pdb.set_trace()

    else:
        #im_names = ['KITTI_000017.png']
        im_names = ['KITTI_003683.png', 'KITTI_003684.png', 'KITTI_003686.png', 'KITTI_003687.png',
       		'KITTI_003690.png', 'KITTI_003694.png', 'KITTI_003709.png' ] # KITTI Test set]
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
            im_name = os.path.join( cfg.DATA_DIR, 'demo', im_name )
            demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
     
    #plt.show()
