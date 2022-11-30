import gc
import os
import pickle as pkl

from lib.conceptors import *

"""
Facade to simplify experiment code, e.g., when computing certain values or caching.
"""

def compute_Cs_and_Ns(group, esn, aperture="auto", normalize=True, XorZ="X", cache=True):
    Cs = compute_Cs(group=group, esn=esn, aperture=aperture, normalize=normalize, XorZ=XorZ, cache=cache)
    print("- computing negative conceptors")
    Ns = Ns_from_Cs(Cs)
    return Cs, Ns


def try_reading_from_cache(file_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_path = current_dir + '/../cache/'

    if os.path.exists(cache_path + file_name + '.pkl'):
        print("- loading from file")
        fp = open(cache_path + file_name + '.pkl', 'rb')
        data = pkl.load(fp)
        fp.close()
        print("--- Done")
        gc.collect()
        return data
    else:
        return None


def save_to_cache(file_name, data):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_path = current_dir + '/../cache/'
    fp = open(cache_path + file_name + '.pkl', 'wb')
    pkl.dump(data, fp)
    fp.close()


def compute_Cs(group=None, signals=None, esn=None, aperture="auto", normalize=True, XorZ="X", cache=True,
               file_identifier=""):
    Cs = False
    if signals is None:
        signals = group.values()

    file_name = file_identifier + XorZ + str(aperture) + str(esn.esn_params) + str(len(list(signals))) + str(
        len(list(signals)[0])) + "ps"
    if cache:
        Cs = try_reading_from_cache(file_name)

    if not Cs:
        print("- computing conceptors")
        Cs = []
        if group is None:
            for signal in signals:
                X = esn.run(signal.T, XorZ=XorZ)
                if aperture == "auto":
                    Cs.append(compute_c(X, 1))
                else:
                    Cs.append(compute_c(X, aperture))
        else:
            for _, signals in group.items():
                X = run_all(esn, signals, XorZ)
                if aperture == "auto":
                    Cs.append(compute_c(X, 1))
                else:
                    Cs.append(compute_c(X, aperture))
        if aperture == "auto":
            print("optimizing")
            Cs = optimize_apertures(Cs, start=0.5, end=500, n=150)
        if normalize:
            print("normalizing")
            Cs = normalize_apertures(Cs)

        if cache:
            save_to_cache(file_name, Cs)

    return Cs


def run_all(esn, signals, XorZ):
    X = np.array([])
    for signal in signals:
        x = esn.run(signal.T, XorZ=XorZ)
        X = np.hstack((X, x)) if X.size else x
    return X
