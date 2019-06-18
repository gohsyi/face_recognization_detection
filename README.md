# Face Recognization and Detection
Course project for SJTU CS385: Machine Learning, taught by Prof. Quanshi Zhang

## Requirement
- numpy
- tensorflow

## Run
Unzip `FDDB-folds.tgz` and `originalPics.tar.gz` into `data/`

Run `python preprocess.py`

To do face recognization, run `python recognization.py` with following arguments
```
-model: model name, lr/svm/fisher/cnn
-lr: learning rate
-total_epoches: total training epoches
-batch_size: batch size for cnn model
-kernel: kernel type of svm
-langevin: if set, use langevin dynamics
-gpu: gpu device
```

To do face detection, run `python detection.py` with following arguments
```
-model: model name, lr/svm/cnn
-winSize: sliding window size
stride: sliding window stride length
thres_score: threshold of score
thres_iou: threshold of IoU
scales: scale factors, use , to seperate
hw_ratios: height-width ratios, use , to seperate
```
