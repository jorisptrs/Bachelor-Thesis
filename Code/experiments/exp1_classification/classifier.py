from dataset.data_processing import group_by_labels
from debug import debug_print
from experiments.helpers.experiment_helpers import *
from sklearn.base import BaseEstimator, ClassifierMixin

from lib.esn import ESN


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

        self.n_mels = None
        self.XorZ = None
        self.N = None
        self.n_samples = None
        self.classes = None
        self.esn = None
        self.normalize = None
        self.Cs_clas = None
        self.Ns_clas = None

    def fit(self, X, y, **params):
        self.n_mels = params["n_mels"]
        self.XorZ = params["XorZ"]
        self.N = params["N"]
        self.cache = params["cache"]
        self.normalize = params["normalize"]

        # Group data by class
        group = group_by_labels(X, y)

        self.classes = list(group.keys())
        self.n_samples = sum([len(x) for x in list(group.values())])

        debug_print(f"Number of samples: {self.n_samples}")
        # Init Reservoir
        esn_params = {
            "in_dim": self.n_mels,
            "out_dim": self.n_mels,
            "N": self.N,
            "W_in_scale": self.W_in_scale,
            "b_scale": self.b_scale,
            "spectral_radius": self.spectral_radius,
            "weights": self.weights
        }
        self.esn = ESN(esn_params)
        self.Cs_clas, self.Ns_clas = compute_Cs_and_Ns(group=group, esn=self.esn, aperture="auto", normalize=self.normalize, XorZ=self.XorZ, cache=self.cache)

        # Return the classifier
        return self

    def predict(self, X):
        y = []
        for sample in X:
            x = self.esn.run(sample.T, XorZ=self.XorZ)
            es = evidences_for_Cs(x,self.Cs_clas,self.Ns_clas)
            if self.XorZ == "X":
                es = [ np.sum(p) for p in es ]
            y.append(self.classes[np.argmax(es)])

        return y