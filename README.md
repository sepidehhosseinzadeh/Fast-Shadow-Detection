# Fast-Shadow-Detection

This code is for the paper S Hosseinzadeh, etc. "Fast Shadow Detection from a Single Image Using a Patched Convolutional Neural Network", Proceedings of the 2018 IEEE/IROS 2018

https://arxiv.org/abs/1709.09283

# Generating the probability map images

Using paper http://dhoiem.cs.illinois.edu/publications/pami12_shadow.pdf

Code: http://aqua.cs.uiuc.edu/site/projects/shadow.html

# Dependencies:
1- nolearn

2- lasagne

3- theano

Python libraries:

4- scipy

5- sklearn

6- matplotlib

7- skimage

8- Python’s basic libraries (pickle, sys, os, urllib, gzip, cPickle, h5py, math, time, pdb)

# How to run the code:

Run commad:

python2 main_fast_shadow_detection.py 

OR

python3 main_fast_shadow_detection_p3.py

# Notes: 

Build folders "data_cache" and "prediction_output_v1" for data training/testing output files, and output prediction result files.

TrainImgeFolder: Training Images

TrainMaskFolder: Training Masks (Ground Truth)

TrainFCNFolder: Probability map images

Likewise for testing images…

The Mask files should have 1 dimension.

# Using GPU:

Content in ~/.theanorc:

[global]

floatX = float32

[nvcc]

fastmath = True
