# My own results
---

## Test time

 With CUDA 8.0 + cuDNN v5.0 (DL-34 machine)

| Models   | Storage | Time per iter |
|:--------:|:-------:|:-------------:|
| ZF       | HDD     | 0.139s        |
| VGG16    | HDD     | 0.282s        |
| ResNet50 | HDD     | 0.628s        |


 With CUDA 8.0 + cuDNN v5.1 (DL-34 machine)

| Models | Storage | Time per iter |
|:------:|:-------:|:-------------:|
| ZF | HDD | 0.139s |
| VGG16 | HDD | 0.279s |
| VGG16 | SSD | 0.277s |
| ResNet50 | SSD | 0.558s |
 