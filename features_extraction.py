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
# from skimage import transform
# from skimage import util
# from skimage import filters

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
X, y = pre.read_dataset(pth='./dataset64/')
print(X.shape, y.shape)

# %%
for x in X:
    x = pre.fix_illumination(pre.fix_contrast(x))

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
def extract_features(img):
    gray_img = color.rgb2gray(img)
    # convert to uint8
    gray_img = (gray_img * 255).astype(np.uint8)
    return np.concatenate((HoG(gray_img), LBP(gray_img).flatten(), GLCM(gray_img).flatten()))

# %%
# Extract features
X_features = np.array([extract_features(x) for x in tqdm.tqdm(X)])
print(X_features.shape)

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# %%
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# %%
# Train model
def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))

# %%
clf = SVC(kernel='linear', C=1.0, random_state=42)
# test_model(clf, X_train, y_train, X_test, y_test)

# %%
# Naive Bayes
clf = GaussianNB()
test_model(clf, X_train, y_train, X_test, y_test)
