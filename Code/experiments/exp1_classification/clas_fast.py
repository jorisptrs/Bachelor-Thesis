import numpy as np
import random
import os, sys
import warnings

from matplotlib import pyplot as plt

from lib.esn import ESN
from dataset.loading import DataLoader

random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

directory = os.path.abspath('/Users/joris/Documents/Work/bsc ai/bt/Bachelor-Thesis/code')
sys.path.append(directory)

data_path = '../../../'
dl = DataLoader(data_path)

dr = []
speakers = []
XorZ = "X"
long_version = False
n_mels = 14
delta = False
delta_delta = False
subsamples = 10
const_params = {
    "n_mels": n_mels,
    "XorZ": XorZ,
    "N": 100
}

path_option = "Final" + str(long_version) + str(n_mels) + str(delta) + str(delta_delta) + str(subsamples)

if dr:
    path_option = str(dr) + "_" + path_option
if len(speakers):
    path_option = str(speakers[0]) + "_" + path_option

features_train, labels_train, _ = dl.collectFeaturesInSegments(
    n_mels=n_mels, delta=delta, delta_delta=delta_delta,
    long_version=long_version, speakers=speakers, dr=dr,
    subsamples=subsamples, path_option=path_option)

from dataset.data_processing import *

phonemes, features_train, labels_train = filter_data(features_train, labels_train, limit=2000)

from experiments.helpers.experiment_helpers import *
from sklearn.base import BaseEstimator, ClassifierMixin


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 W_in_scale,
                 b_scale,
                 spectral_radius,
                 weights):
        self.W_in_scale = W_in_scale
        self.spectral_radius = spectral_radius
        self.b_scale = b_scale
        self.weights = weights

    def fit(self, X, y, **params):
        self.n_mels = params["n_mels"]
        self.XorZ = params["XorZ"]

        # Group data by class
        group = group_by_labels(X, y)

        self.classes = list(group.keys())
        self.n_samples = sum([len(x) for x in list(group.values())])

        print(f"Number of samples: {self.n_samples}")
        # Init Reservoir
        esn_params = {
            "in_dim": self.n_mels,
            "out_dim": self.n_mels,
            "N": 100,
            "W_in_scale": self.W_in_scale,
            "b_scale": self.b_scale,
            "spectral_radius": self.spectral_radius,
            "weights": self.weights
        }
        self.esn = ESN(esn_params)
        self.Cs_clas, self.Ns_clas = compute_Cs_and_Ns(group, esn=self.esn, aperture="auto", normalize=True,
                                                       XorZ=self.XorZ, cache=False)

        # Return the classifier
        return self

    def predict(self, X):
        y = []
        for sample in X:
            x = self.esn.run(sample.T, XorZ=self.XorZ)
            es = evidences_for_Cs(x, self.Cs_clas, self.Ns_clas)
            if self.XorZ == "X":
                es = [np.sum(p) for p in es]
            y.append(self.classes[np.argmax(es)])

        return y
# Old params:
#opt = Classifier(1.5, .2, 1.5, .1) # Z optimal params

#clas = Classifier(1.1, .44, 2.57, .1)

clas = Classifier(
    1.1, .8, 2.57, .1
)

clas.fit(features_train, labels_train, **{
    "n_mels":n_mels,
    "XorZ":'X',
    "N": 100
})

# Method "Z
# optz = Classifier(1.5,
#     .2,
#     1.5,
#     .1)
#
# optz.fit(features_train, labels_train, **{
#     "n_mels":n_mels,
#     "XorZ":"Z"
# })

# Scikitlearn


#opt = Classifier()
#opt = BayesSearchCV(Classifier(), parameters, n_iter=50, cv=3)
#opt.fit(features, labels, **{
#    "in_dim":n_mels,
#    "out_dim":n_mels
#})
#print(opt.score(fv, lv))
#print(opt.best_params_)

print("Testing...")

features_test, labels_test, _ = dl.collectFeaturesInSegments(
    ft='Test',n_mels=n_mels,delta=delta,delta_delta=delta_delta,
    long_version=long_version,speakers=[],dr=dr,sentence=[],
    subsamples=subsamples,path_option=path_option+"_test")

_, features_test, labels_test = filter_data(features_test, labels_test, limit=None)

print(f'Training Accuracy : {clas.score(features_train, labels_train)}')
print(f'Test Accuracy     : {clas.score(features_test, labels_test)}')
