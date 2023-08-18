import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import os
import sys

path_feats = os.path.join('..', 'data', 'feats')
path_tuning = os.path.join('..', 'data', 'tuning')
if not os.path.exists(path_tuning):
    os.makedirs(path_tuning)

cnns = []
cnns.append('densenet121')
cnns.append('densenet169')
cnns.append('densenet201')
cnns.append('efficientnetb0')
cnns.append('efficientnetb1')
cnns.append('efficientnetb2')
cnns.append('efficientnetb3')
cnns.append('efficientnetb4')
cnns.append('efficientnetb5')
cnns.append('efficientnetb6')
cnns.append('efficientnetb7')
cnns.append('inceptionresnetv2')
cnns.append('inceptionv3')
cnns.append('mobilenet')
cnns.append('mobilenetv2')
cnns.append('mobilenetv3large')
cnns.append('mobilenetv3small')
cnns.append('nasnetlarge')
cnns.append('nasnetmobile')
cnns.append('resnet101')
cnns.append('resnet101v2')
cnns.append('resnet152')
cnns.append('resnet152v2')
cnns.append('resnet50')
cnns.append('resnet50v2')
cnns.append('vgg16')
cnns.append('vgg19')
cnns.append('xception')

methods = ['KNN', 'MLR', 'RF', 'SVM']

for method in methods:

    if method == 'KNN':
        hypers = [1, 3, 5, 7, 9]
        name_hyper = 'K'
    elif method == 'MLR':
        hypers = [1e-2, 1e-1, 1, 10, 100]
        name_hyper = '\u03B1'
    elif method == 'RF':
        hypers = [0.2, 0.4, 0.6, 0.8, 1]
        name_hyper = '\u03B7'
    else:
        hypers = [1e-2, 1e-1, 1, 10, 100]
        name_hyper = 'C'
   
    accs = np.zeros((len(cnns), len(hypers), 10))
    
    for hyper_idx, hyper in enumerate(hypers):

        if method == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=hyper)
        elif method == 'MLR':
            clf = LogisticRegression(C=hyper, random_state=5489)
        elif method == 'RF':
            clf = RandomForestClassifier(n_estimators=200, max_features=hyper, n_jobs=-1, random_state=5489)
        else:
            clf = SVC(kernel='linear', C=hyper, random_state=5489)

        scaler = MinMaxScaler()

        for cnn_idx, cnn in enumerate(cnns):

            path_cnn = os.path.join(path_feats, cnn)

            for fold_idx in range(1, 11):

                path_train = os.path.join(path_cnn, 'train_{:02d}'.format(fold_idx))
                mat_train = loadmat(os.path.join(path_train, 'data.mat'))
                X_train = mat_train['features']
                Y_train = mat_train['labels']
                y_train = Y_train.argmax(axis=1)
                path_valid = os.path.join(path_cnn, 'valid_{:02d}'.format(fold_idx))
                mat_valid = loadmat(os.path.join(path_valid, 'data.mat'))
                X_valid = mat_valid['features']
                Y_valid = mat_valid['labels']
                y_valid = Y_valid.argmax(axis=1)
                scaler.fit(X_train)
                X_train_normalized = scaler.transform(X_train)
                X_valid_normalized = scaler.transform(X_valid)
                clf.fit(X_train_normalized, y_train)
                y_pred = clf.predict(X_valid_normalized)
                acc = accuracy_score(y_valid, y_pred)
                accs[cnn_idx, hyper_idx, fold_idx-1] = acc

                print('method = {}, {} = {:g}, cnn = {}, fold = {:02d}: acc = {:.2f}'.format(method, name_hyper, hyper, cnn, fold_idx, acc))
    
    ranks = np.zeros(accs.shape)
    for i in range(10):
        ranks[:,:,i] = np.reshape(rankdata(1-accs[:,:,i], method='average'), (len(cnns), len(hypers)))
    
    avg_ranks = ranks.mean(axis=2)
    df = pd.DataFrame(avg_ranks, index=cnns, columns=hypers)
    latex_table = df.to_latex()
   
    # Saving results in Latex
    with open(os.path.join(path_tuning, f'{method}.tex'), 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Average rankings for different settings of the {method} classifier.}}\n')
        f.write(latex_table)
        f.write(f'\\label{{tab:{method}}}\n')
        f.write('\\end{table}\n')
