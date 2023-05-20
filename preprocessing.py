# %% [markdown]
# # preprocess images for flood detection task

# %%
import os
import tqdm
from skimage import io
import numpy as np
import shutil
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# %% [markdown]
# # Ideas:
#     - normalize images shape
#     - normalize colors
#     - noise?
#     - illumination?
#     - contrast?

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


# %%



