# %% [markdown]
#  # preprocess images for flood detection task

# %%
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
# noise
from skimage.filters import median
# illumination
# contrast
from skimage import exposure

# %% [markdown]
#  # Ideas:
#      - normalize images shape
#      - normalize colors
#      - noise?
#      - illumination?
#      - contrast?

# %%
def resize_img(img, W, H):
    img = resize(img, (W, H))
    return img

# %%
def read_resize_dataset(pth='./dataset/', pth2=None, W=None, H=None, Debug=False):
    # read images
    pth = './dataset/'
    if W is None and H is None:
        W, H = 64, 64
    if W is not None or H is not None:
        if W is None: W = H
        if H is None: H = W
    if pth2 is None:
        pth2 = pth[:-1] + str(W) + 'x' + str(H) + '/'

    if Debug:
        print(f'from {pth} to {pth2}, with size {W}x{H}')

    # create new folders
    if not Debug and not os.path.exists(pth2):
        os.mkdir(pth2)
        os.mkdir(pth2+'flooded')
        os.mkdir(pth2+'non-flooded')

    for fil in os.listdir(pth + 'flooded/'):
        # already done
        if os.path.exists(pth2+'flooded/'+fil):
            continue
        img = io.imread(pth+'flooded/'+fil)
        if Debug:
            io.imshow(img)
            io.show()
        # resize image
        img = resize(img, (W, H))
        if Debug:
            io.imshow(img)
            io.show()
            break
        # convert to uint
        img = img.astype(np.uint8)
        # save image
        io.imsave(pth2+'flooded/'+fil, img)

    for fil in os.listdir(pth + 'non-flooded/'):
        # already done
        if os.path.exists(pth2+'non-flooded/'+fil):
            continue
        img = io.imread(pth+'non-flooded/'+fil)
        if Debug:
            io.imshow(img)
            io.show()
        # resize image
        img = resize(img, (W, H))
        if Debug:
            io.imshow(img)
            io.show()
            break
        # convert to uint
        # img = img.astype(np.uint8) # produces some black images
        # save image
        io.imsave(pth2+'non-flooded/'+fil, img)


# %%
def read_dataset(pth='./dataset/'):
    # read images
    flooded = []
    non_flooded = []
    for fil in os.listdir(pth + 'flooded/'):
        img = io.imread(pth+'flooded/'+fil)
        flooded.append(img)
    for fil in os.listdir(pth + 'non-flooded/'):
        img = io.imread(pth+'non-flooded/'+fil)
        non_flooded.append(img)
    #
    X = np.array(flooded + non_flooded)
    y = np.array([1]*len(flooded) + [0]*len(non_flooded))
    return X, y

# %%
## more preprocessing
# noise
def remove_noise(img):
    img2 = img.copy()
    # use median filter from skimage
    img2 = median(img2)
    return img2

# %%
# illumintaion
def fix_illumination(img):
    img2 = img.copy()
    # use skimage
    img2 = exposure.equalize_adapthist(img2, clip_limit=0.03)
    return img2

# %%
# contrast
def fix_contrast(img):
    img2 = img.copy()
    # use skimage
    img2 = exposure.equalize_hist(img2)
    return img2

# %%
def test_preprocessing():
    tst = io.imread('./dataset/flooded/0.jpg')

    fig = plt.figure(figsize=(17, 7))

    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(tst)
    ax.set_title('Original')

    ax = fig.add_subplot(2, 4, 2)
    img2 = fix_contrast(tst)
    ax.imshow(img2)
    ax.set_title('Contrast')

    ax = fig.add_subplot(2, 4, 3)
    img2 = remove_noise(tst)
    ax.imshow(img2)
    ax.set_title('Noise')

    ax = fig.add_subplot(2, 4, 4)
    img2 = fix_illumination(tst)
    ax.imshow(img2)
    ax.set_title('Illumination')

    ax = fig.add_subplot(2, 4, 5)
    img2 = fix_contrast(tst)
    img2 = fix_illumination(img2)
    ax.imshow(img2)
    ax.set_title('Contrast + Illumination')

    plt.show()

# %%
# read_resize_dataset(W=64)

# X, y = read_dataset(pth='./dataset64/')
# print(X.shape, y.shape)

# test_preprocessing()


