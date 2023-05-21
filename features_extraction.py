# %% [markdown]
#  # preprocess images for flood detection task

# %%
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
# HoG
# LBP
# GLCM
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import exposure
# from skimage import transform
# from skimage import util

import preprocessing as pre

# %% [markdown]
#  # Observations:
#  
#    - no water -> no flood LOL
#    - no buildings nor road -> no flood
#    - wrecked buildings -> no flood
#    - 
#    - cars -> flood
#    - weird waterline -> flood
#    - too much water -> flood
#    - very high position of camera -> flood

# %% [markdown]
# # Suggested Features to capture observations:
#   - HoG
#   - GLCM
#   - LBP
#   - Color Features in all 3 channels ?
#   - Waterline using edge detection (Canny, Sobel, etc)
#   - Waterline using color detection (HSV, RGB, etc)

# %%
def HoG(img):
    return feature.hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)

# %%
def LBP(img):
    return feature.local_binary_pattern(img, 8, 1, method='uniform')

# %%
def GLCM(img):
    return feature.graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

# %%
# Color Histogram
def color_histogram(img_hsv):
    hist, bins = exposure.histogram(img_hsv[:, :, 0], nbins=180)
    return hist

# %%
# Color Moments
def color_moments(img_hsv):
    # use measure
    moments = measure.moments(img_hsv[:, :, 0], order=3)
    central_moments = measure.moments_central(moments)
    normalized_moments = measure.moments_normalized(central_moments)
    hu_moments = measure.moments_hu(normalized_moments)
    return hu_moments

# %%
# watershed
def watershed(img):
    return filters.rank.gradient(img, morphology.disk(5))

# %%
# Canny
def canny(img):
    return feature.canny(img, sigma=3)

# %%
# Sobel
def sobel(img):
    return filters.sobel(img)

# %%
def extract_features(img, HOG=False, LBP=False, GLCM=False, color_histogram=False, color_moments=False, watershed=False, canny=False, sobel=False):
    gray_img = color.rgb2gray(img)
    # convert to uint8
    gray_img = (gray_img * 255).astype(np.uint8)
    # convert to HSV
    img_hsv = color.rgb2hsv(img)
    #
    features = np.array([])
    if HOG:
        features = np.concatenate((features, HoG(gray_img)))
    if LBP:
        features = np.concatenate((features, LBP(gray_img).flatten()))
    if GLCM:
        features = np.concatenate((features, GLCM(gray_img).flatten()))
    if color_histogram:
        features = np.concatenate((features, color_histogram(img_hsv).flatten()))
    if color_moments:
        features = np.concatenate((features, color_moments(img_hsv).flatten()))
    if watershed:
        features = np.concatenate((features, watershed(gray_img).flatten()))
    if canny:
        features = np.concatenate((features, canny(gray_img).flatten()))
    if sobel:
        features = np.concatenate((features, sobel(gray_img).flatten()))
    #
    return features


