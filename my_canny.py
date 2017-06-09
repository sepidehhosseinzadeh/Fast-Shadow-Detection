import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import misc
from skimage import feature

def canny_edge(im, sig=3, low_th=12, high_th=80):
    imgray = np.mean(im, axis = 2);
    edges = feature.canny(imgray, sigma=sig, low_threshold=low_th, high_threshold=high_th);
    return edges;

def canny_edge_on_mask(imgray):
    edges = feature.canny(imgray, sigma=1.5, low_threshold=12, high_threshold=80);
    return edges;

