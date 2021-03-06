#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  kaist)
    DB_NAME="kaist"
    TRAIN_IMDB="kaist_2015_train02"
    TEST_IMDB="kaist_2015_test20"
    PT_DIR="kaist"
    CONFIG="experiments/cfgs/faster_rcnn_end2end_kaist.yml"    
    ITERS=150000
    ;;
  kitti_trainval)
    DB_NAME="kitti_all"
    TRAIN_IMDB="kitti_2012_trainval"
    TEST_IMDB="kitti_2012_val"
    PT_DIR="kitti"
    CONFIG="experiments/cfgs/faster_rcnn_end2end_kitti.yml"
    ITERS=100000
    ;;
  kitti)
    DB_NAME="kitti"
    TRAIN_IMDB="kitti_2012_train"
    TEST_IMDB="kitti_2012_val"
    PT_DIR="kitti"
    CONFIG="experiments/cfgs/faster_rcnn_end2end_kitti.yml"
    #ITERS=450000   # For AlexNet
    ITERS=150000
    ;;
  voc_0712)
    DB_NAME="voc0712"
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2012_test"
    PT_DIR="pascal_voc"
    CONFIG="experiments/cfgs/faster_rcnn_end2end.yml"
    ITERS=100000
    ;;
  pascal_voc)
    DB_NAME="voc07"
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    CONFIG="experiments/cfgs/faster_rcnn_end2end.yml"
    ITERS=70000
    ;;
  coco14_trainval)
    DB_NAME="coco14_all"
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train+coco_2014_val"
    TEST_IMDB="coco_2015_test"
    PT_DIR="coco14_trainval"
    CONFIG="experiments/cfgs/faster_rcnn_end2end.yml"
    ITERS=70000
    ;;
  coco)
    DB_NAME="coco"
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    CONFIG="experiments/cfgs/faster_rcnn_end2end.yml"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${DB_NAME}_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb_train ${TRAIN_IMDB} \
  --imdb_val ${TEST_IMDB} \
  --iters ${ITERS} \
  --cfg ${CONFIG} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x


case $DATASET in
  kitti)
    time ./tools/eval_kitti.py --gpu ${GPU_ID} \
      --net ${NET} \
      --iter ${ITERS}
  *)
    time ./tools/test_net.py --gpu ${GPU_ID} \
      --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
      --net ${NET_FINAL} \
      --imdb ${TEST_IMDB} \
      --cfg ${CONFIG} \
      ${EXTRA_ARGS}