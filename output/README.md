# My own results


## Training time

 With CUDA 8.0 + cuDNN v5.0 (DL-34 machine)

| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| ZF       | HDD     | 0.139s        |
| VGG16    | HDD     | 0.282s        |
| ResNet50 | HDD     | 0.628s        |


 With CUDA 8.0 + cuDNN v5.1 (DL-34 machine)

| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| ZF       | HDD     | 0.139s        |
| VGG16    | HDD     | 0.279s        |
| VGG16    | SSD     | 0.277s        |
| ResNet50 | SSD     | 0.558s        |


## Test time
| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| VGG16    | SSD     | 0.072s        |
| ResNet50 | SSD     | 0.135s        |

## Accuracy
| Models    | Mean AP | Training        | Testing     |
|:---------:|:-------:|:---------------:|:-----------:| 
| VGG16     | 0.691   | VOC 07 trainval | VOC 07 test |
| ResNet50  | 0.723   | VOC 07 trainval | VOC 07 test | 


## Accuracy in detail

| Models | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow | giningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG16 | 0.691 | 0.791 | 0.672 | 0.567 | 0.529 | 0.771 | 0.805 | 0.793 | 0.481 | 0.769 | 0.643 | 0.770 | 0.805 | 0.767 | 0.772 | 0.433 | 0.664 | 0.647 |0.758 | 0.683 | 0.691 |
| ResNet50 | 0.742 | 0.797 | 0.689 | 0.592 | 0.526 | 0.828 | 0.796 | 0.861 | 0.519 | 0.761 | 0.682 | 0.833 | 0.806 | 0.785 | 0.781 | 0.459 | 0.736 | 0.744 | 0.781 | 0.738 | 0.723 |   
 