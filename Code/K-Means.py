from json.encoder import INFINITY
from math import floor
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg
inv = linalg.inv

from lib.conceptors import *
from lib.esn import ESN

"""
Problem-specific plotting
"""
class Plot:
    def __init__(self):
        self.ymax = 3
        self.xmax = 3
        self.fig, self.ax = plt.subplots(self.ymax,self.xmax)
        self.cnt = 0

    def add(self, y, label="No Label"):
        y_idx = self.cnt % self.ymax
        x_idx = int(floor(self.cnt / self.ymax))
        self.ax[y_idx, x_idx].plot(y, label=label)
        self.ax[y_idx, x_idx].legend()

    def inc(self):
        if self.cnt < self.xmax * self.ymax:
            self.cnt += 1

    def add_new(self, y, label="No Label"):
        self.inc()
        self.add(y, label)

    def add_new_assignment_plot(self,assignments):
        self.inc()
        for i in assignments:
            self.add(i)

    def finalize(self, title="No Title"):
        self.fig.suptitle(title, fontsize=16)
        plt.show()

    def conceptors_fit_plot(self, X, Cs, label):
        self.inc()
        collection, _ = test(X, Cs, "PROP")
        for i, vals in enumerate(collection):
            # walking average of d
            d = 20
            smoothed = np.convolve(np.array(vals), np.ones(d), 'valid') / d
            self.add(smoothed, label=label+str(i))

plot = Plot()

########################################
# Hyperparameters

# np.random.seed(0)
t_max = 1000
t_test = 50
t_washout = 500 # number of washout steps
aperture = 5

esn_params = {
    "in_dim": 1,
    "out_dim": 1,
    "N": 100,
    "W_in_scale": 1.5,
    "b_scale": .2,
    "spectral_radius": 1.5
}

########################################
# collect data
def gen_signal(n, period, amplitude):
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1/period) * ts)
    return data

print("Generating signals")
p1 = gen_signal(t_max, 8, 1)
p2 = gen_signal(t_max, 14, 1)
p3 = gen_signal(t_max, 20, 1)
data = [p1, p2, p3]
# plot.add(p1[:t_test], "p1")
# plot.add_new(p2[:t_test], "p2")
# plot.add_new(p3[:t_test], "p3")
p_combined = np.concatenate((p1,p2[t_washout:],p3[t_washout:]))
plot.add(p_combined, label="Input signal")
#######################################
# init reservoir

esn = ESN(esn_params)

#######################################
# run the reservoir with the signal(s) and collect X
print("Running and loading reservoir with different signals")
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

X_combined, X_combined_delay = esn.run(p_combined, t_washout)
p_combined = p_combined[t_washout:]
plot.conceptors_fit_plot(X_combined, original_Cs, "Original C")

#######################################
esn.load(X_combined, X_combined_delay, 1e-4)
esn.train_out_identity(X_combined, p_combined, 1e-2)

#######################################
# Generate mixed signal

def assign_to_clusters(nb_points, nb_clusters, mode="EQUAL_SPLIT", limits=[]):
    assignments = [ [] for _ in range(nb_clusters) ]
    if mode == "RANDOM" or mode == "EQUAL_SPLIT":
        points = [ x for x in range(nb_points) ]
        if mode == "RANDOM":
            np.random.shuffle(points)
        for i in range(nb_clusters):
            assignments[i] = points[i*int(nb_points/nb_clusters):(i+1)*int(nb_points/nb_clusters)]
    elif mode == "RANGES":
        mark = 0
        for i in range(nb_points):
            if i in limits:
                mark += 1
            assignments[mark].append(i)
    return assignments

def assign_fuzzy_to_clusters(nb_points, nb_clusters, transition_time):
    """
    Distributes points over clusters smoothly changing float membership
    """
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
    return assignments

assignments = assign_fuzzy_to_clusters(t_max, 3, transition_time=100)
y, X_regen = esn.generate(original_Cs, assignments, t_max, True)
#plot.add_new(y, label="Generated signal")

#######################################
# K-means

def kmeans(X, nb_conceptors, aperture, epochs=100):
    print("K-means")
    # Initial assignments and initial conceptors
    nb_points = X.shape[1]
    new_assignments = assign_to_clusters(nb_points, nb_conceptors, "RANDOM")
    #new_assignments = assign_to_clusters(nb_points, nb_conceptors, "RANGES", [500,800])
    #new_assignments = assign_to_clusters(nb_points, nb_conceptors, "EQUAL_SPLIT")
    # Training loop
    for epoch in range(epochs):
        print("epoch:",epoch)
        # recompute centroids based on subset of assigned state
        Cs = [ compute_c(X[:,assignments], aperture) for assignments in new_assignments ]
        plot.conceptors_fit_plot(X, Cs, "K-means epoch:"+str(epoch)+", C")
        # recompute assignments by find the closest conceptor for each of the state points
        old_assignments = new_assignments.copy()
        new_assignments = [ [] for _ in range(nb_conceptors) ]
        for t in range(nb_points):
            conceptor_index, _ = find_closest_C(X[:,t], Cs)
            new_assignments[ conceptor_index ].append(t)

        # stop if converged
        for new_assignment, old_assignment in zip(new_assignments, old_assignments):
            if set(new_assignment) == set(old_assignment):
                print("Converged")
                return Cs, new_assignments

    return Cs, new_assignments

# cluster into as many conceptors as patterns
kmeans_Cs, assignments = kmeans(X_combined, nb_conceptors=3, aperture=aperture, epochs=100)

#######################################
# Classify on original and clustered conceptors
#plot.conceptors_fit_plot(X_combined, kmeans_Cs, "Kmeans C")

plot.finalize(title='Aperture='+str(aperture)+', N='+str(esn.N)+', Spec Rad='+str(esn.spectral_radius))
