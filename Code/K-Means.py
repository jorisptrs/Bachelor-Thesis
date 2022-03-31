from json.encoder import INFINITY
from math import floor
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg
inv = linalg.inv

from lib.conceptors import *
from lib.esn import ESN

def shift(signal, phase):
    for _ in range(phase):
        signal = signal[-1] + signal[:-1]
    return signal

def ridge_regression(X, Y, reg_param):
    """
    Ridge regression solving W x = y
    """
    XTX = X.T @ X
    return (np.linalg.inv(XTX + reg_param * np.eye(XTX.shape[0])) @ X.T @ Y).T

#######################################


# Evaluate

class Plot:
    def __init__(self):
        self.ymax = 3
        self.xmax = 3
        self.fig, self.ax = plt.subplots(self.ymax,self.xmax)
        self.cnt = 0

    def add(self, y, label="No Label"):
        y_idx = self.cnt % self.ymax
        x_idx = int(floor(self.cnt / self.ymax))
        self.ax[y_idx,x_idx].plot(y, label=label)
        self.ax[y_idx,x_idx].legend()
        if label == "Original C":
            print("OG C", y_idx, x_idx)

    def inc(self):
        if self.cnt < self.xmax * self.ymax:
            self.cnt += 1

    def addNew(self, y, label="No Label"):
        self.inc()
        self.add(y, label)

    def finalize(self, title="No Title"):
        plot.fig.suptitle(title, fontsize=16)
        plt.show()

    def conceptors_fit_plot(self, X, Cs, label):
        collection, _ = test(X, Cs, "PROP")
        plot.inc()
        for i, vals in enumerate(collection):
            # walking average of d
            d = 15
            smoothed = np.convolve(np.array(vals), np.ones(d), 'valid') / d
            plot.add(smoothed, label=label+str(i))

def predict_correct():
    for i, iterations in enumerate(assignments):
        i = 0
        original_perf += mean_fit(X_regen[:,i:i+iterations], original_Cs, i) / len(y)
        i += iterations

plot = Plot()

# np.random.seed(0)
t_max = 1500
t_test = 50
t_washout = 500 # number of washout steps
aperture = 5

########################################
# collect data
def gen_signal(n, period, amplitude):
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1/period) * ts)
    return data

p1 = gen_signal(t_max, 5, 1)
p2 = gen_signal(t_max, 10, 1)
p3 = gen_signal(t_max, 15, 1)
data = [p1, p2, p3]
plot.add(p1[:t_test], "p1")
plot.addNew(p2[:t_test], "p2")
plot.addNew(p3[:t_test], "p3")

#######################################

# init reservoir
esn = ESN()

#######################################
# run the reservoir with the signal(s) and collect X
X = None
X_delay = None
P = None
original_Cs = []
for signal in data:
    X_local, X_delay_local = esn.run(signal, t_washout)
    if X is None:
        X = X_local
        X_delay = X_delay_local
        P = signal[t_washout:]
    else:
        # append new states X_local, X_delay (later used for loading), and pattern
        X = np.concatenate((X, X_local), axis=1)
        X_delay = np.concatenate((X_delay, X_delay_local), axis=1)
        P = np.concatenate((P, signal[t_washout:]))
    original_Cs.append( compute_c(X_local, aperture) )

plot.conceptors_fit_plot(X, original_Cs, "Original C")

#######################################
esn.load(X, X_delay, 1e-4)
esn.train_out_identity(X, P, 1e-2)

#######################################
# Generate mixed signal

def assign_to_clusters(nb_points, nb_clusters, mode="EQUAL_SPLIT", limits=[]):
    assignments = [ [] for _ in range(nb_clusters) ]
    if mode == "RANDOM" or mode == "EQUAL_SPLIT":
        assignments = [ x for x in range(nb_points) ]
        if mode == "RANDOM":
            np.random.shuffle(assignments)
        for i in range(nb_clusters):
            assignments[i] = assignments[i*int(nb_points/nb_clusters):(i+1)*int(nb_points/nb_clusters)]
    elif mode == "RANGES":
        mark = 0
        for i in range(nb_points):
            if i in limits:
                mark += 1
            assignments[mark].append(i)
    return assignments


def assign_fuzzy_to_clusters(nb_points, nb_clusters, transition_time):
    assignments = [ [] for _ in range(nb_clusters) ]
    mean_length = int(nb_points/nb_clusters)
    current = -1
    for t in range(nb_points):
        if not t % mean_length:
            current += 1
        for i in range(nb_clusters):
            if i == current:
                if t % mean_length > mean_length - transition_time and not i + 1 == nb_clusters:
                    assignments[i].append(1 - (t % mean_length - (mean_length - transition_time)) * 1 / transition_time)
                else:
                    assignments[i].append(1)
            elif i == current + 1 and t % mean_length > mean_length - transition_time:
                assignments[i].append((t % mean_length - (mean_length - transition_time)) * 1 / transition_time)
            else:
                assignments[i].append(0)
    # plot.inc()
    # for i in assignments:
    #     plot.add(i)
    # plot.finalize()
    return assignments


assignments = assign_fuzzy_to_clusters(1500, 3, 50) #assign_to_clusters(1500, 3)
y, X_regen = esn.generate(original_Cs, assignments, 1500, True)
plot.addNew(y, label="Generated signal")
plot.finalize()


#######################################
# K-means



def kmeans(y_length, nb_conceptors):
    # Initial assignments and initial conceptors
    new_assignments = assign_to_clusters(y_length, nb_conceptors, "RANDOM")
    # Training loop
    for epoch in range(100):
        print("epoch:",epoch)
        # recompute centroids based on subset of assigned state
        conceptors = [ compute_c(X_regen[:,assignments], aperture) for assignments in new_assignments ]
        # recompute assignments by find the closest conceptor for each of the state points
        old_assignments = new_assignments.copy()
        new_assignments = [ [] for _ in range(nb_conceptors) ]
        for t in range(y_length):
            conceptor_index, _ = find_closest_C(X_regen[:,t], conceptors)
            new_assignments[ conceptor_index ].append(t)

        # stop if converged
        for new_assignment, old_assignment in zip(new_assignments, old_assignments):
            if set(new_assignment) == set(old_assignment):
                print("Converged")
                return conceptors, new_assignments

    return conceptors, new_assignments

kmeans_Cs, assignments = kmeans(len(y), 3) # as many conceptors as loaded patterns

#######################################
# Classify on original and clustered conceptors
plot.conceptors_fit_plot(X, kmeans_Cs, "Kmeans C")

plot.finalize(title='Aperture='+str(aperture)+', N='+str(esn.N)+', Spec Rad='+str(esn.spectral_radius))
