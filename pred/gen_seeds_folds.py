import csv
import cv2
import os
from PIL import Image
import sys

path_folds = os.path.join('..', 'data', 'folds')
path_seeds = os.path.join('..', 'data', 'seeds')

folds = [fold.split('.')[0] for fold in sorted(os.listdir(path_folds))]
cultivars = [cultivar for cultivar in sorted(os.listdir(path_seeds))]

dst_dir_base = os.path.join('..', 'data', 'seeds_folds')
if not os.path.exists(dst_dir_base):
    os.makedirs(dst_dir_base)

for fold in folds:
    dst_dir = os.path.join(dst_dir_base, fold)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for cultivar in cultivars:
        dst_dir = os.path.join(dst_dir_base, fold, cultivar)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

for fold in folds:
    with open(os.path.join(path_folds, fold + '.csv'), mode='r') as csvfile:
        reader = csv.reader(csvfile)
        samples = list(reader)
        for sample in samples:
            cultivar, seed = sample
            print(f'In fold {fold}, saving the seed {seed} from cultivar {cultivar}...')
            img_bgr = cv2.imread(os.path.join(path_seeds, cultivar, seed + '.png'))
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img, 'RGB')
            pil.save(os.path.join(dst_dir_base, fold, cultivar, seed + '.png'))
print('Done!')
