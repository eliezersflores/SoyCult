import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold 

path_seeds_base = os.path.join('..', 'data', 'seeds')
path_folds = os.path.join('..', 'data', 'folds')
if not os.path.exists(path_folds):
    os.makedirs(path_folds)

cultivars = sorted(os.listdir(path_seeds_base))

names = []
labels = []

for i, cultivar in enumerate(cultivars):
    seeds_dir = os.path.join(path_seeds_base, cultivar)
    names_cultivar = [[cultivar, fname.split('.png')[0]] for fname in sorted(os.listdir(seeds_dir))]
    names += names_cultivar
    labels += [i]*len(names_cultivar)

names = np.array(names)
labels = np.array(labels)

Xtrain, Xtest, ytrain, ytest = train_test_split(names, labels, 
                                                    test_size=0.2, 
                                                    stratify=labels,
                                                    random_state=5489)

print('Saving a CSV file with the test images list...')
with open(os.path.join(path_folds, 'test.csv'), mode='w') as file_test:
    writer_test = csv.writer(file_test)
    writer_test.writerows(Xtest)

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=5489)

for fold_idx, (train_idxs, valid_idxs) in enumerate(cv.split(Xtrain, ytrain), start=1):
    Xtrain_fold, Xvalid_fold = Xtrain[train_idxs], Xtrain[valid_idxs]
    fname_train = 'train_{:02d}.csv'.format(fold_idx)
    fname_valid = 'valid_{:02d}.csv'.format(fold_idx)
    print(f'Saving a CSV file with the train images list for the fold {fold_idx}...')
    with open(os.path.join(path_folds, fname_train), mode='w') as file_train:
        writer_train = csv.writer(file_train)
        writer_train.writerows(Xtrain_fold)
    print(f'Saving a CSV file with the valid images list for the fold {fold_idx}...')
    with open(os.path.join(path_folds, fname_valid), mode='w') as file_valid:
        writer_valid = csv.writer(file_valid)
        writer_valid.writerows(Xvalid_fold)
print('Done!')
