import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(
    'c:\\main\\Work\\thesis\\Bachelor-Thesis\\Code\\hierarchical-clustering\\hierarchical_clustering.py'))))
import matplotlib.pyplot as plt
import pickle as pkl
import gc
from timit.loading import Feature_Collector
from lib.conceptors import *
from lib.helpers import *
import warnings

warnings.filterwarnings("ignore")

###-------------- Collecting Training Features -------------------
path = '../timit/'
fc = Feature_Collector(path)

save = True

dr = []
speakers = []
long_version = False
n_mels = 15
delta = False
delta_delta = False
subsamples = 10

path_option = str(long_version) + "_" + str(n_mels) + "_" + str(delta) + "_" + str(delta_delta) + "_" + str(subsamples)
if dr:
    path_option = str(dr) + "_" + path_option
if speakers:
    path_option = str(speakers) + "_" + path_option

if save and os.path.exists('./cache/working/' + path_option + '_features.pkl'):
    if os.path.exists('./cache/working/' + path_option + '_labels.pkl'):
        print("-from output")
        ffp = open('./cache/working/' + path_option + '_features.pkl', 'rb')
        flp = open('./cache/working/' + path_option + '_labels.pkl', 'rb')
    features = pkl.load(ffp)
    labels = pkl.load(flp)
    ffp.close()
    flp.close()
    print('---- success')
    # -------
else:
    print('--- Failed')
    print('Collecting Features from Audio Files')
    features, labels, oversamplings = fc.collectFeaturesInSegments(
        n_mels=n_mels, delta=delta, delta_delta=delta_delta,
        long_version=long_version, speakers=speakers, dr=dr,
        subsamples=subsamples)
    print("Oversamplings: ", oversamplings)
    # -------------
    if save:
        ffp = open("./cache/working/" + path_option + "_features.pkl", 'wb')
        pkl.dump(features, ffp)
        flp = open("./cache/working/" + path_option + "_labels.pkl", 'wb')
        pkl.dump(labels, flp)
        ffp.close()
        flp.close()
    print('--- Completed')
    # -------
gc.collect()
