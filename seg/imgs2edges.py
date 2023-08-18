import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from webcolors import name_to_rgb

#**********************************************************
# METHOD PARAMETERS
thresh1 = 150
thresh2 = 300
# PRINT PARAMETERS
color_name = 'magenta'
color_code = tuple(name_to_rgb(color_name))
thickness = 4 
#**********************************************************

imgs_dir = os.path.join('..', 'data', 'imgs')
edges_dir = os.path.join('..', 'data', 'edges')
if not os.path.exists(edges_dir):
    os.makedirs(edges_dir)
overlays_dir = os.path.join('..', 'data', 'edges_overlayed')
if not os.path.exists(overlays_dir):
    os.makedirs(overlays_dir)

names = [fname.split('.')[0] for fname in os.listdir(imgs_dir) if fname.endswith('.JPG')]

for name in names:

    print(f'Computing the edges of the image {name}...')

    img_bgr = cv2.imread(os.path.join(imgs_dir, name + '.JPG'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    edges_r = cv2.Canny(img[:,:,0], thresh1, thresh2)
    edges_g = cv2.Canny(img[:,:,1], thresh1, thresh2)
    edges_b = cv2.Canny(img[:,:,2], thresh1, thresh2)
    edges = cv2.bitwise_or(edges_r, edges_g)
    edges = cv2.bitwise_or(edges, edges_b)

    pil_edges = Image.fromarray(edges, 'L')
    pil_edges.save(os.path.join(edges_dir, name + '.png'))

    locs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, locs, -1, color_code, thickness=thickness)

    pil_overlay = Image.fromarray(overlay, 'RGB')
    pil_overlay.save(os.path.join(overlays_dir, name + '.png'))

print('Done!')
