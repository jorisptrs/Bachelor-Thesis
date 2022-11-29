import gc
import os
import pickle as pkl

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

from lib.conceptors import *
from lib.plot import Plot


def compute_Cs_and_Ns(group, esn, aperture="auto", normalize=True, XorZ="X", cache=True):
    Cs = compute_Cs(group, esn, aperture, normalize, XorZ, cache)
    print("- computing negative conceptors")
    Ns = Ns_from_Cs(Cs)
    return Cs, Ns


def try_reading_from_cache(file_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_path = current_dir + '/../cache/'

    if os.path.exists(cache_path + file_name + '.pkl'):
        print("- loading conceptors from file")
        fp = open(cache_path + file_name + '.pkl', 'rb')
        data = pkl.load(fp)
        fp.close()
        print("--- Done")
        gc.collect()
        return data
    else:
        return False


def save_to_cache(file_name, data):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_path = current_dir + '/../cache/'
    fp = open(cache_path + file_name + '.pkl', 'wb')
    pkl.dump(data, fp)
    fp.close()


def compute_Cs(group=None, signals=None, esn=None, aperture="auto", normalize=True, XorZ="X", cache=True, file_identifier=""):
    Cs = False
    if signals is None:
        signals = group.values()
    if cache:
        file_name = file_identifier + XorZ + str(aperture) + str(esn.esn_params) + str(len(list(signals))) + str(
            len(list(signals)[0])) + "ps"
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


def compute_Cs_from_X_list(X_list, aperture="auto"):
    Cs = False
    if cache:
        file_name = file_identifier + XorZ + str(aperture) + str(esn.esn_params) + str(len(list(group.keys()))) + str(
            len(list(group.values())[0])) + "ps"
        Cs = try_reading_from_cache(file_name)
    if not Cs:
        print("- computing conceptors")
        Cs = []
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

# by changing the coorinates of the above you can repeat this for the y axis too
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def run_all(esn, signals, XorZ):
    X = np.array([])
    for signal in signals:
        x = esn.run(signal.T, XorZ=XorZ)
        X = np.hstack((X, x)) if X.size else x
    return X
