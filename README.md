# Dual Channel Residual Network for Hyperspectral Image Classification with Noisy Labels
A python deep learning model which is super useful in dealing with hyperspectral image classification with noisy labels.
# Description
The loss function gets its code from [Link](https://github.com/HanxunH/Active-Passive-Losses)
# Requirement
This code is compatible with Python 3.7.6.

It is based on PyTorch 1.5.1 and torchvision 0.6.1

If a GPU is detected, CUDA 10.0+ is used to boost training.
# How to Run
## Run with default settings
Default settings are capsuled in main.py already. If tested with default parameters, just run main.py like **python3 main.py**
## Run with alternative settings
Currently, we only support a few adjustable hyper-parameters:
- batch_size: batch size, default to be 16
- max_iter: max training epochs, default to be 100
- iters: repetition experments' number, default to be 10
- lr: learning rate, default to be 0.001
Example to run with personal settings: **python3 main.py --batch_size 32 --max_iter 200**
# Citing this work
