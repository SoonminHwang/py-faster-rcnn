# My own results


## Training time
[DL-34] Machine spec: Ubuntu 14.04 LTS, NVidia Titan X (Pascal) - Driver 367.57 

 With CUDA 8.0 + cuDNN v5.0

| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| ZF       | HDD     | 0.139s        |
| VGG16    | HDD     | 0.282s        |
| ResNet50 | HDD     | 0.628s        |


 With CUDA 8.0 + cuDNN v5.1

| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| ZF       | HDD     | 0.139s        |
| VGG16    | HDD     | 0.279s        |
| VGG16    | SSD     | 0.277s        |
| ResNet50 | SSD     | 0.558s        |


## Test time
| Models   | Storage | Time per frame | Dataset |
|:--------:|:-------:|:--------------:|:-------:|
| VGG16    | SSD     | 0.072s         | VOC 07  |
| ResNet50 | SSD     | 0.135s         | VOC 07  |
| ZF       | SSD     | 0.031s         | VOC 07  |
| VGG16    | SSD     | -              | COCO 15 |
| VGG16    | SSD     | 0.056s         | KITTI   |


## Accuracy
| Num | Models    | Mean AP | Training           | Testing      |
|:---:|:---------:|:-------:|:------------------:|:------------:| 
| 1   | ZF        | 0.603   | VOC 07 trainval    | VOC 07 test  |
| 1-1 | ZF        | 0.597   | VOC 07 trainval    | VOC 07 test  |
| 2   | VGG16     | 0.691   | VOC 07 trainval    | VOC 07 test  |
| 3   | ResNet50  | 0.723   | VOC 07 trainval    | VOC 07 test  |
| 4   | ZF        | -       | VOC 07+12 trainval | VOC 12 test  | 
| 5   | VGG16     | -       | VOC 07+12 trainval | VOC 12 test  |
| 6   | ResNet50  | -       | VOC 07+12 trainval | VOC 12 test  |
| 7   | ZF        | -       | COCO 14 trainval   | COCO 15 test | 
| 8   | VGG16     | -       | COCO 14 trainval   | COCO 15 test |
| 9   | VGG16     | 0.469   | KITTI train        | KITTI val    |
| 10  | ResNet50  | -       | KITTI train        | KITTI val    |
| 11  | VGG16     | 0.469   | KITTI train        | KITTI val    |

## Accuracy in detail

VOC 07 test

| Num | Models | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow | giningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor | mAP |
|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1   | ZF    | 0.628 | 0.712 | 0.546 | 0.457 | 0.344 | 0.671 | 0.742 | 0.730 | 0.385 | 0.667 | 0.626 | 0.662 | 0.752 | 0.661 | 0.672 | 0.349 | 0.585 | 0.530 | 0.707 | 0.625 | 0.603 |
| 1-1 | ZF    | 0.628 | 0.725 | 0.549 | 0.418 | 0.357 | 0.663 | 0.774 | 0.714 | 0.380 | 0.667 | 0.604 | 0.648 | 0.765 | 0.663 | 0.669 | 0.321 | 0.584 | 0.534 | 0.690 | 0.625 | 0.597 |
| 2   | VGG16 | 0.691 | 0.791 | 0.672 | 0.567 | 0.529 | 0.771 | 0.805 | 0.793 | 0.481 | 0.769 | 0.643 | 0.770 | 0.805 | 0.767 | 0.772 | 0.433 | 0.664 | 0.647 |0.758 | 0.683 | 0.691 |
| 3   | ResNet50 | 0.742 | 0.797 | 0.689 | 0.592 | 0.526 | 0.828 | 0.796 | 0.861 | 0.519 | 0.761 | 0.682 | 0.833 | 0.806 | 0.785 | 0.781 | 0.459 | 0.736 | 0.744 | 0.781 | 0.738 | 0.723 |   



KITTI val

| Num | Models    | Pedestrian | Cyclist | Car   | All   |
|:---:|:---------:|:----------:|:-------:|:-----:|:-----:|
| 9   | VGG16     | 32.8       | 23.7    | 28.0  | 46.9  |
| 10  | ResNet50  | 30.1       | 20.7    | 25.4  | 44.1  |
| 11  | VGG16     | 32.6       | 23.3    | 27.6  | 46.9  |

## Tips
 * If you don't want to terminate a process while you logged out for ssh session, use ```nohup``` command and ```&```.
 * Example) ```nohup ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc & ```