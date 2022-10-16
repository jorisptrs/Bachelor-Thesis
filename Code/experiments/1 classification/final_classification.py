import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(0)
import random

random.seed(0)
import os, sys

directory = os.path.abspath('/Users/joris/Documents/Work/bsc ai/thesis/bachelor-thesis/code/')
sys.path.append(directory)

from data.loading import Feature_Collector
from lib.conceptors import *
from lib.esn import ESN
from lib.helpers import *
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

import warnings

warnings.filterwarnings("ignore")

XorZ = "X"
# method = "centroids"
# method = "sims"
method = "pred"
# method = "ogsignals"

print("starting")

path = '../timit/'
fc = Feature_Collector(path)

dr = []
speakers = []

long_version = False
n_mels = 13
delta = False
delta_delta = False
subsamples = 10

# Number of cross validations during optimization
cv = 2
# Number of iterations during optimization
n_iter = 2

path_option = "Final" + str(long_version) + str(n_mels) + str(delta) + str(delta_delta) + str(subsamples)

if dr:
    path_option = str(dr) + "_" + path_option
if len(speakers):
    path_option = str(speakers[0]) + "_" + path_option

features_train, labels_train, _ = fc.collectFeaturesInSegments(
    n_mels=n_mels, delta=delta, delta_delta=delta_delta,
    long_version=long_version, speakers=speakers, dr=dr,
    subsamples=subsamples, path_option=path_option)

gc.collect()


#
# Regroup data and subset phonemes
#

def filter_data(unfiltered_features, unfiltered_labels, limit=None):
    group = {}

    for i in range(len(unfiltered_labels)):
        if unfiltered_labels[i] not in group.keys():
            group[unfiltered_labels[i]] = []
        group[unfiltered_labels[i]].append(unfiltered_features[i])

    for key in group.keys():
        if limit is not None and len(group[key]) > limit:
            group[key] = random.sample(group[key], limit)

    classes = list(group.keys())
    filtered_labels = []
    filtered_features = []
    for label, features in group.items():
        for feature in features:
            filtered_features.append(feature)
            filtered_labels.append(label)
    return classes, filtered_features, filtered_labels


phonemes, features_train, labels_train = filter_data(features_train, labels_train, limit=None)

print(f"Filtered to {len(features_train)} samples of shape {features_train[0].shape}")


#
# Compute Conceptors
#

def compute_Cs_and_Ns(group, esn, initial_aperture):
    Cs = []
    for phoneme, signals in group.items():
        X = np.array([])
        for signal in signals:
            x = esn.run(signal.T, XorZ=XorZ)
            X = np.hstack((X, x)) if X.size else x
        Cs.append(compute_c(X, initial_aperture))
    print("optimizing")
    Cs = optimize_apertures(Cs)
    print("normalizing")
    Cs, _ = normalize_apertures(Cs)
    print("- computing negative conceptors")
    Ns = Ns_from_Cs(Cs)
    return Cs, Ns


from sklearn.base import BaseEstimator, ClassifierMixin


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, W_in_scale=1.1, spectral_radius=2.57, b_scale=.44, weights=.1):
        self.esn = None
        self.Cs_clas = None
        self.Ns_clas = None
        self.classes = None
        self.n_samples = None
        self.W_in_scale = W_in_scale
        self.spectral_radius = spectral_radius
        self.b_scale = b_scale
        self.weights = weights

    def fit(self, X, y, **params):
        # Group data by class
        group = {}
        for i in range(len(y)):
            if y[i] not in group.keys():
                group[y[i]] = []
            group[y[i]].append(X[i])

        self.classes = list(group.keys())
        self.n_samples = sum([len(x) for x in list(group.values())])

        print("Number of samples:", self.n_samples)
        print("Classes:", self.classes)

        # Init Reservoir

        esn_params = {
            "in_dim": params["in_dim"],
            "out_dim": params["out_dim"],
            "N": 100,
            "W_in_scale": self.W_in_scale,
            "b_scale": self.b_scale,
            "spectral_radius": self.spectral_radius,
            "weights": self.weights
        }

        self.esn = ESN(esn_params)
        self.Cs_clas, self.Ns_clas = compute_Cs_and_Ns(group, esn=self.esn, initial_aperture=1)

        # Return the classifier
        return self

    def predict(self, X):
        y = []
        for sample in X:
            if XorZ == "X":
                x, _ = self.esn.run_X(sample.T, 0, sample.shape[0])
                es = evidences_for_Cs(x, self.Cs_clas, self.Ns_clas)
                es = [np.sum(p) for p in es]
            else:
                z = self.esn.run(sample.T)
                es = evidences_for_Cs(z, self.Cs_clas, self.Ns_clas)
            y.append(self.classes[np.argmax(es)])
        return y


parameters = {
    'W_in_scale': Real(0.01, 2),
    'spectral_radius': Real(0.01, 4),
    'b_scale': Real(0, 2),
    'weights': Real(0.01, 0.99)
}

# opt = Classifier()
opt = Classifier(1.1, 2.57, .44, .1)
# opt = BayesSearchCV(Classifier(), parameters, n_iter=n_iter, cv=cv)
opt.fit(features_train, labels_train, **{
    "in_dim": n_mels,
    "out_dim": n_mels
})

print("Optimal parameters:")
# for key, value in opt.best_params_.items():
#    print(key, ':', value)

# TODO
# Plot progress
#pd.DataFrame(opt.cv_results_).plot(figsize=(10,10))
#plt.show()

print("Testing...")

features_test, labels_test, _ = fc.collectFeaturesInSegments(
    ft='Test', n_mels=n_mels, delta=delta, delta_delta=delta_delta,
    long_version=long_version, speakers=[], dr=dr, sentence=[],
    subsamples=subsamples, path_option=path_option + "_test")

print(f"{len(features_test)} test features")
_, features_test, labels_test = filter_data(features_test, labels_test, limit=1000000)

print(f'Test Accuracy     : {opt.score(features_test, labels_test)}')
print(f'Training Accuracy : {opt.score(features_train, labels_train)}')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)

pred_test = opt.predict(features_test)
# Plot non-normalized confusion matrix
plot_confusion_matrix(labels_test, pred_test, classes=phonemes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(labels_test, pred_test, classes=phonemes,
                      normalize=True,
                      title='Normalized confusion matrix')

plt.show()

with open('res.txt', 'x') as file:
    file.write(str(test_score))
    # file.write(str(opt.best_params_))








