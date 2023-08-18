import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys

conts_dir = os.path.join('..', 'data', 'conts')
holes_dir = os.path.join('..', 'data', 'holes')
if not os.path.exists(holes_dir):
    os.makedirs(holes_dir)

names = [fname.split('.')[0] for fname in os.listdir(conts_dir) if fname.endswith('.png')]

for name in names:

    print(f'Obtaining the holes from the contours image {name}...')

    conts = cv2.imread(os.path.join(conts_dir, name + '.png'), cv2.IMREAD_GRAYSCALE)
    locs, _ = cv2.findContours(conts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = np.zeros_like(conts)
    cv2.drawContours(holes, locs, -1, (255), cv2.FILLED)

    pil_conts = Image.fromarray(holes.astype('uint8'), 'L')
    pil_conts.save(os.path.join(holes_dir, name + '.png'))

print('Done!')
