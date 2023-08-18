import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy import ndimage
from skimage import measure
import sys

#**********************************************************
# METHOD PARAMETERS
max_width = 126
max_height = 123 
#**********************************************************

def bounding_box(binary_img):
    rows, cols = np.nonzero(binary_img)
    top = np.min(rows)
    bottom = np.max(rows)
    left = np.min(cols)
    right = np.max(cols)
    return (left, top, right - left + 1, bottom - top + 1)

def remove_small_holes(binary_img, min_area):
    labels = measure.label(binary_img)
    props = measure.regionprops(labels)
    mask = np.zeros_like(binary_img, dtype=bool)
    for prop in props:
        if prop.area >= min_area:
            mask = np.logical_or(mask, labels == prop.label)
    filtered = binary_img*mask
    return filtered

imgs_dir = os.path.join('..', 'data', 'imgs')
holes_dir = os.path.join('..', 'data', 'holes')

seeds_base_dir = os.path.join('..', 'data', 'seeds')
if not os.path.exists(seeds_base_dir):
    os.makedirs(seeds_base_dir)

names = [fname.split('.')[0] for fname in os.listdir(imgs_dir) if fname.endswith('.JPG')]

for name in names:

    seeds_dir = os.path.join(seeds_base_dir, name)
    if not os.path.exists(seeds_dir):
        os.makedirs(seeds_dir)

    img_bgr = cv2.imread(os.path.join(imgs_dir, name + '.JPG'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    nr, nc, nb = img.shape

    holes = cv2.imread(os.path.join(holes_dir, name + '.png'), cv2.IMREAD_GRAYSCALE)/255
    filtered = np.multiply(img, np.expand_dims(holes, axis=2))

    labels, nl = ndimage.label(holes)

    for label in range(1, nl + 1):

        print(f'In the image {name}, extracting the seed n. {label} of {nl} ...')

        background = np.zeros((nr, nc))
        background[labels==label] = 1
        left, top, width, height = bounding_box(background)

        mask = background[top:top+height-1, left:left+width-1]
        filtered_mask = remove_small_holes(mask, 700)

        seed = filtered[top:top+height-1, left:left+width-1, :]
        filtered_seed = np.multiply(seed, np.expand_dims(mask, axis=2))

        padded_seed = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        padded_seed[max_height//2-height//2:max_height//2-height//2+height-1,
                max_width//2-width//2:max_width//2-width//2+width-1] = filtered_seed

        pil = Image.fromarray(padded_seed, 'RGB')
        pil.save(os.path.join(seeds_dir, '{:03d}'.format(label) + '.png'))
