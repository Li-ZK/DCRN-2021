# Dual Channel Residual Network for Hyperspectral Image Classification with Noisy Labels
A python deep learning model which is super useful in dealing with hyperspectral image classification with noisy labels.

# Description
The loss function gets its code from [Link](https://github.com/HanxunH/Active-Passive-Losses)

Links to relevant comparison methods are shown below.
* [SSRN  Spectral-spatial residualnetwork for hyperspectral image classification: A 3-D deep learning framework](https://github.com/zilongzhong/SSRN)
* [DPNLD  Density Peak-based Noisy Label Detection for Hyperspectral Image Classification] (https://github.com/xf-zh/Density-Peak-based-Noisy-Label-Detection-for-Hyperspectral-Image-Classification)
* [RLPA   Hyperspectral image classification in the presence of noisy labels] (https://github.com/junjun-jiang/RLPA)
* [K-SDP  Spatial density peak clustering for hyperspectral image classification with noisy labels] (https://github.com/Li-ZK/DCRN-2021/SDP)
*[3D-CNN  Spectral–spatial classification of hyperspectral imagery with 3D convolutional neural network](https://github.com/mhaut/hyperspectral_deeplearning_review/tree/master/algorithms)

# Requirement
This code is compatible with Python 3.7.6.

It is based on PyTorch 1.5.1 and torchvision 0.6.1

If a GPU is detected, CUDA 10.0+ is used to boost training.
# Usage
### Run with default settings
Default settings are capsuled in main.py already. If tested with default parameters, just run main.py like

``` python3 main.py```

### Run with alternative settings
Currently, we only support a few adjustable hyper-parameters:
- batch_size: batch size, default to be 16
- max_iter: max training epochs, default to be 100
- iters: repetition experments' number, default to be 10
- lr: learning rate, default to be 0.001.

Example to run with personal settings: 

```python3 main.py --batch_size 32 --max_iter 200```

## Exemplar dataset folder
The code in SDP is for paper "Spatial Density Peak Clustering of Hyperspectral Images with Noise Labels".

An example dataset folder has the following structure:
```
datasets
├── KSC
│   ├── KSC.mat
│   ├── KSC_gt.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```

# Citing this work