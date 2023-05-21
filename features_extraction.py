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
def getHoG(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False):
    return feature.hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize)

# %%
def getLBP(img_gray, radius=3, n_points=None):
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# %%
def getGLCM(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True, CONTRAST=True, CORRELATION=True, DISSIMILARITY=False, HOMOGENEITY=False, ENERGY=False, ASM=False):
    glcm = feature.graycomatrix(img_gray, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    #
    features = np.array([])
    if CONTRAST:
        features = np.concatenate((features, feature.graycoprops(glcm, 'contrast').ravel()))
    if CORRELATION:
        features = np.concatenate((features, feature.graycoprops(glcm, 'correlation').ravel()))
    if DISSIMILARITY:
        features = np.concatenate((features, feature.graycoprops(glcm, 'dissimilarity').ravel()))
    if HOMOGENEITY:
        features = np.concatenate((features, feature.graycoprops(glcm, 'homogeneity').ravel()))
    if ENERGY:
        features = np.concatenate((features, feature.graycoprops(glcm, 'energy').ravel()))
    if ASM:
        features = np.concatenate((features, feature.graycoprops(glcm, 'ASM').ravel()))
    return features

# %%
# Color Histogram
def color_histogram(img_hsv, bins=16):
    hist, bins = exposure.histogram(img_hsv[:, :, 0], nbins=bins)
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# %%
# Color Moments
def color_moments(img_lab):
    means = np.mean(img_lab, axis=(0, 1))
    stds = np.std(img_lab, axis=(0, 1))
    return np.concatenate((means, stds))

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
def extract_features(img, HOG=False, LBP=False, GLCM=False, COLOR_HISTOGRAM=False, COLOR_MOMENTS=False, WATERSHED=False, CANNY=False, SOBEL=False):
    gray_img = color.rgb2gray(img)
    # convert to uint8
    gray_img = (gray_img * 255).astype(np.uint8)
    # convert to HSV
    img_hsv = color.rgb2hsv(img)
    #
    features = np.array([])
    if HOG:
        features = np.concatenate((features, getHoG(gray_img)))
    if LBP:
        features = np.concatenate((features, getLBP(gray_img).flatten()))
    if GLCM:
        features = np.concatenate((features, getGLCM(gray_img).flatten()))
    if COLOR_HISTOGRAM:
        features = np.concatenate((features, color_histogram(img_hsv).flatten()))
    if COLOR_MOMENTS:
        features = np.concatenate((features, color_moments(img_hsv).flatten()))
    if WATERSHED:
        features = np.concatenate((features, watershed(gray_img).flatten()))
    if CANNY:
        features = np.concatenate((features, canny(gray_img).flatten()))
    if SOBEL:
        features = np.concatenate((features, sobel(gray_img).flatten()))
    #
    return features