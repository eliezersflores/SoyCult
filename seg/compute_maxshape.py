import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import sys

def bounding_box(binary_image):
    rows, cols = np.nonzero(binary_image)
    top = np.min(rows)
    bottom = np.max(rows)
    left = np.min(cols)
    right = np.max(cols)
    return (left, top, right - left + 1, bottom - top + 1)

holes_dir = os.path.join('.', 'data', 'holes')

names = [fname.split('.')[0] for fname in os.listdir(holes_dir) if fname.endswith('.png')]

max_width = -np.inf
max_height = -np.inf

for name in names:

    holes = cv2.imread(os.path.join(holes_dir, name + '.png'), cv2.IMREAD_GRAYSCALE)
    nr, nc = holes.shape

    labels, nl = ndimage.label(holes)

    for label in range(1, nl + 1):
        print(f'In the image {name}, computing the dimensions of the seed n. {label} of {nl}...')
        background = np.zeros((nr, nc))
        background[labels==label] = 1
        left, top, width, height = bounding_box(background)
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
        
print(f'=> Maximum width across all the images = {max_width}')
print(f'=> Maximum height across all the images = {max_height}')
