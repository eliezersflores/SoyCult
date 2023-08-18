import matplotlib.pyplot as plt
import numpy as np
from Orange.evaluation import compute_CD, graph_ranks
import os
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import chi2, friedmanchisquare, rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#**********************************************************
# PRINT PARAMETERS
fontsize1 = 16
fontsize2 = 20
plt.rcParams['font.family'] = 'serif'
#**********************************************************

methods = ['knn', 'lda', 'mlr', 'nb', 'rf', 'svm']

path_feats = os.path.join('..', 'data', 'feats', 'densenet201')
path_seeds = os.path.join('..', 'data', 'seeds')

path_results = os.path.join('..', 'data', 'results')
if not os.path.exists(path_results):
    os.makedirs(path_results)

path_conf_matrices = os.path.join(path_results, 'conf_matrices')
if not os.path.exists(path_conf_matrices):
    os.makedirs(path_conf_matrices)

path_post_hoc = os.path.join(path_results, 'post_hoc')
if not os.path.exists(path_post_hoc):
    os.makedirs(path_post_hoc)

cultivars = sorted(os.listdir(path_seeds))

path_train = os.path.join(path_feats, 'train_01')
path_valid = os.path.join(path_feats, 'valid_01')
path_test = os.path.join(path_feats, 'test')

mat_train = loadmat(os.path.join(path_train, 'data.mat'))
mat_valid = loadmat(os.path.join(path_valid, 'data.mat'))
mat_test = loadmat(os.path.join(path_test, 'data.mat'))

X_train = np.concatenate((mat_train['features'], mat_valid['features']), axis=0)
Y_train = np.concatenate((mat_train['labels'], mat_valid['labels']), axis=0)
y_train = Y_train.argmax(axis=1)

X_test = mat_test['features']
Y_test = mat_test['labels']
y_test = Y_test.argmax(axis=1)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)

performances_array = np.empty((len(cultivars), len(methods)))

for i, method in enumerate(methods):

    print(f'Testing with the {method} method...')

    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=9)
    elif method == 'lda':
        clf = LinearDiscriminantAnalysis()
    elif method == 'mlr':
        clf = LogisticRegression(C=0.2, random_state=5489)
    elif method == 'nb':
        clf = GaussianNB()
    elif method == 'rf':
        clf = RandomForestClassifier(n_estimators=200, max_features=1, n_jobs=-1, random_state=5489)
    else:
        clf = SVC(kernel='linear', C=0.01, random_state=5489)

    clf.fit(X_train_normalized, y_train)
    y_pred = clf.predict(X_test_normalized)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Plotting and saving the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cultivars, yticklabels=cultivars, ax=ax, annot_kws={'size': fontsize1})
    ax.set_xlabel('Cultivar (predicted)', fontsize=fontsize2)
    ax.set_ylabel('Cultivar (actual)', fontsize=fontsize2)
    ax.set_xticklabels(cultivars, rotation=45, ha='right', fontsize=fontsize2)
    ax.set_yticklabels(cultivars, rotation=0, fontsize=fontsize2)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize2)
    plt.tight_layout()
    plt.savefig(os.path.join(path_conf_matrices, f'{method}.pdf'), format='pdf')

    # Computing per class accuracy
    totals = np.sum(cm, axis=1)
    accs = np.diag(cm)/totals
    performances_array[:, i] = accs

print('Done!')

# Performing the Friedman chi-square test
statistic, p_value = friedmanchisquare(*performances_array.T)
print("Friedman Chi-square statistic:", statistic)
print("p-value:", p_value)

# Performing the Nemenyi post-hoc test
ranks = np.array([rankdata(-p) for p in performances_array])
average_ranks = np.mean(ranks, axis=0)
cd = compute_CD(average_ranks,
n=len(performances_array),
alpha='0.05',
test='nemenyi')

# Plotting and saving the visualization of the Nemenyi post-hoc test results.
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.linewidth'] = 10
graph_ranks(
        filename='nemenyi_post.pdf',
        avranks=average_ranks,
        names=['KNN', 'LDA', 'MLR', 'NB', 'RF', 'SVM'],
        cd=cd,
        width=6,
        textspace=1,
        reverse=False
)
plt.savefig(os.path.join(path_post_hoc, 'nemenyi_cdplot.pdf'), format='pdf', dpi=300)

