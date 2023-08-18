import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from webcolors import name_to_rgb

#**********************************************************
# METHOD PARAMETERS
thresh = 75 
# PRINT PARAMETERS
color_name = 'magenta'
color_code = tuple(name_to_rgb(color_name))
thickness = 4 
#**********************************************************

imgs_dir = os.path.join('..', 'data', 'imgs')
edges_dir = os.path.join('..', 'data', 'edges')
conts_dir = os.path.join('..', 'data', 'conts')
if not os.path.exists(conts_dir):
    os.makedirs(conts_dir)
overlays_dir = os.path.join('..', 'data', 'conts_overlayed')
if not os.path.exists(overlays_dir):
    os.makedirs(overlays_dir)

names = [fname.split('.')[0] for fname in os.listdir(edges_dir) if fname.endswith('.png')]

for name in names:

    print(f'Obtaining the contours from the edges image {name}...')

    img_bgr = cv2.imread(os.path.join(imgs_dir, name + '.JPG'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    edges = cv2.imread(os.path.join(edges_dir, name + '.png'), cv2.IMREAD_GRAYSCALE)
    locs_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = np.zeros_like(edges)
    for loc in locs_edges:
        area = cv2.contourArea(loc)
        if area >= thresh:
            ellipse = cv2.fitEllipse(loc)
            cv2.ellipse(conts, ellipse, 255, 2)

    pil_conts = Image.fromarray(conts.astype('uint8'), 'L')
    pil_conts.save(os.path.join(conts_dir, name + '.png'))

    locs_conts, _ = cv2.findContours(conts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, locs_conts, -1, color_code, thickness=thickness)

    pil_overlay = Image.fromarray(overlay, 'RGB')
    pil_overlay.save(os.path.join(overlays_dir, name + '.png'))

print('Done!')
