# Only a condensed version of the classification.ipynb used for faster
# execution for hyperpamarameter optimization.

# ---------------------------------------------------------------------

import numpy as np
import random
import os, sys
import warnings

from classifier import Classifier
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

# Old params:
#opt = Classifier(1.5, .2, 1.5, .1) # Z optimal params

#clas = Classifier(1.1, .44, 2.57, .1)

clas = Classifier(
    1.1, .8, 2.57, .1
)

clas.fit(features_train, labels_train, **{
    "n_mels":n_mels,
    "XorZ":'X',
    "N": 100,
    "cache": False
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
