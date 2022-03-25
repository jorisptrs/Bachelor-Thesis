from cmath import inf
from json.encoder import INFINITY
from typing import no_type_check
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg, sparse, interpolate

# numpy.linalg is also an option for even fewer dependencies

fig, ax = plt.subplots(4, 2)

# np.random.seed(0)
t_max = 1500
t_test = 50
t_washout = 500 # number of washout steps
L = max(0, t_max - t_washout) # steps after washout
aperture = 5
in_dim = out_dim = 1
N = 100 # reservoir size

########################################
# collect data
def gen_signal(n, period, amplitude):
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1/period) * ts)
    return data
p1 = gen_signal(t_max, 5, 1)
p2 = gen_signal(t_max, 10, 1)
data = [p1, p2]
ax[0,0].plot(p1[:t_test], label="p1")
ax[1,0].plot(p2[:t_test], label="p1")

#######################################
# init reservoir
W_in = 1.5 * np.random.normal(0, 1, (N,in_dim))
b = 0.2 * np.random.normal(0, 1, (N,1))
W_star = sparse.random(N, N, density=.1).toarray()
W_out = None

#######################################
# set the spectral radius
spectral_radius_old = np.max(np.abs(np.linalg.eigvals(W_star)))
spectral_radius = 1.5
W_star *= spectral_radius / spectral_radius_old

#######################################
# conceptors
def compute_conceptor(X, aperture):
    R = np.dot( X, X.T ) / X.shape[1]
    Eig_vals, Eig_vecs = np.linalg.eig(R)
    U = Eig_vecs
    Sigma = np.diag(Eig_vals)
    return np.dot( R, np.linalg.inv( R + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )

#######################################
# run the reservoir with the signal(s) and collect X
Cs = []
X = None
X_delay = None
P = None
for signal in data:
    x = np.random.normal(0, 1, (N,1))
    X_local = np.zeros((N,L))
    X_delay_local = np.zeros((N,L))
    for t in range(t_max):
        if (t >= t_washout):
            X_delay_local[:,t-t_washout] = x[:,0]
        p = signal[t]
        x = np.tanh( np.dot( W_star, x ) + np.dot( W_in, p) + b )
        if (t >= t_washout):
            X_local[:,t-t_washout] = x[:,0]

    if X is None:
        X = X_local
        X_delay = X_delay_local
        P = signal[t_washout:]
    else:
        # append new states X_local, X_delay (later used for loading), and pattern
        X = np.concatenate((X, X_local), axis=1)
        X_delay = np.concatenate((X_delay, X_delay_local), axis=1)
        P = np.concatenate((P, signal[t_washout:]))
    Cs.append( compute_conceptor(X_local, aperture) )

#######################################
# load reservoir by updating W and W_out by ridge regression
reg_W = 1e-4  # regularization coefficient for internal weights
reg_out = 1e-2  # regularization coefficient for output weights
# p (in_dim x t_max)
# W_in (N x (1 + in_dim))
# X (N x L)
# W_star, W (N x N)
# W_out (out_dim x N)
B = np.tile( b, L * len(data))
W = np.dot( np.dot( np.linalg.inv( np.dot( X_delay, X_delay.T ) + reg_W*np.eye(N) ), X_delay ), ( np.arctanh(X)-B ).T ).T
W_out = ( np.dot( np.dot( np.linalg.inv( np.dot( X, X.T ) + reg_out*np.eye(N) ), X), P.T ) ).T

def shift(signal, phase):
    for _ in range(phase):
        signal = signal[-1] + signal[:-1]
    return signal

#######################################
# Generate mixed signal
def generate_mixed_signal(times):
    y_length = 0
    t_transition = 10
    current = 0
    for _, iterations in times:
        y_length += iterations
    
    x = np.random.normal(0, 1, (N,1))
    X_regen = np.zeros((N,y_length)) # collection matrix
    y = np.zeros(y_length)
    t = 0
    # Regenerate signal
    for signal, t_times in times:
        # if signal == current:
        #     for t in range(t_transition):
        #         C = (1 - 1 / t_transition * t) * Cs[current] + (1 / t_transition * t) * Cs[signal]
        #         x = np.dot( C, np.tanh( np.reshape(b,(N,1)) + np.dot( W, x ) ) )
        #         y[t] = np.dot( W_out, x )
        # else:
        for _ in range(t_times):
            x = np.dot( Cs[signal], np.tanh( np.reshape(b,(N,1)) + np.dot( W, x ) ) )
            X_regen[:,t] = x[:,0]
            y[t] = np.dot( W_out, x )
            t += 1
    
    return y, X_regen

times = [ (0,500), (1,500), (0,500) ]
y, X_regen = generate_mixed_signal([ (0,500), (1,500), (0,500) ])
ax[2,0].plot(y, label="Generated signal")

#######################################

# Evaluation
def dist_point_conceptor(point, C):
    dist = np.dot( np.array(point).T, np.dot(C, point) )
    return dist

def summed_distance(X, C):
    sum = 0
    for state_point in zip(*X):
        sum += dist_point_conceptor(state_point, C)
    return sum

#######################################
# K-means

def kmeans(y_length, nb_conceptors):
    # Initial assignments and initial conceptors
    #assignments = np.arange(y_length)
    #np.random.shuffle(assignments)
    assignments = [ x for x in range(y_length) ]
    #indices_by_conceptors = [ assignments[:int(y_length/2)], assignments[int(y_length/2)+1:] ]
    indices_by_conceptors = [ assignments[:500] + assignments[1000:], assignments[500:1000] ]
    new_indices_by_conceptors = indices_by_conceptors.copy()
    # Training loop
    for epoch in range(10000):
        # recompute centroids based on subset of assigned state
        conceptors = [ compute_conceptor(X_regen[:,indices_by_conceptor], aperture) for indices_by_conceptor in new_indices_by_conceptors ]
        # recompute assignments by find the closest conceptor for each of the state points
        old_indices_by_conceptors = new_indices_by_conceptors.copy()
        new_indices_by_conceptors = [ [] for _ in range(nb_conceptors) ]
        for t in range(y_length):
            min_dist = INFINITY
            min_conceptor_index = np.random.choice(range(nb_conceptors))
            for conceptor_index, conceptor in enumerate(conceptors):
                dist = dist_point_conceptor(X_regen[:,t],conceptor) # compute distance from X_regen(:,i) to conceptor
                if dist < min_dist:
                    min_dist = dist
                    min_conceptor_index = conceptor_index
            new_indices_by_conceptors[ min_conceptor_index ].append(t)
        print("epoch=",epoch) 

        # stop if converged
        for new_indices_by_conceptor, old_indices_by_conceptor in zip(new_indices_by_conceptors, old_indices_by_conceptors):
            if new_indices_by_conceptor.sort() == old_indices_by_conceptor.sort():
                return conceptors, new_indices_by_conceptors

conceptors, indices_by_conceptors = kmeans(len(y), len(data)) # as many conceptors as loaded patterns

# Plot memberships
collection = [ [] for _ in range(len(data)) ]
for t in range(1500):
    for idx, Cl in enumerate(indices_by_conceptors):
        if t in Cl:
            collection[idx].append(1)
        else:
            collection[idx].append(0)

ax[3,0].plot(collection[1][:100], color="blue", label="C1")
ax[3,0].plot(collection[0][:100], color="red", label="C1")


#######################################
# Evaluate
kmeans_perf = 0
original_perf = 0

print(len(indices_by_conceptors[0]),len(indices_by_conceptors[1]))

for conceptor, indices_by_conceptor in zip(conceptors, indices_by_conceptors):
    kmeans_perf += summed_distance(X_regen[:,indices_by_conceptor], conceptor) / len(y)

for conceptor_index, iterations in times:
    i = 0
    original_perf += summed_distance(X_regen[:,i:i+iterations], Cs[conceptor_index]) / len(y)
    i += iterations

print("kmeans_perf=",kmeans_perf)
print("original_perf=",original_perf)

fig.suptitle('Aperture='+str(aperture)+', N='+str(N)+', Spec Rad='+str(spectral_radius), fontsize=16)
plt.show()
