from cmath import inf
from json.encoder import INFINITY
from typing import no_type_check
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg, sparse, interpolate
inv = linalg.inv

# numpy.linalg is also an option for even fewer dependencies

fig, ax = plt.subplots(5, 2)

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

p1 = gen_signal(t_max, 5, 3)
p2 = gen_signal(t_max, 10, 1)
p3 = gen_signal(t_max, 15, 5)
data = [p1, p2, p3]
ax[0,0].plot(p1[:t_test], label="p1")
ax[1,0].plot(p2[:t_test], label="p2")
ax[2,0].plot(p3[:t_test], label="p3")

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

def corr(X):
    L = X.shape[1] or 1
    return np.dot( X, X.T ) / L

# conceptors
def compute_c(X, aperture):
    R = corr(X)
    # Eig_vals, Eig_vecs = np.linalg.eig(R)
    # U = Eig_vecs
    # Sigma = np.diag(Eig_vals)
    # for elem in Eig_vals:
    #     if abs(elem) < 1e-100:
    #         print("!!! Zero singular value(s) !!!")
    return np.dot( R, inv( R + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )

def compute_not_c(X, aperture):
    R = corr(X)
    return np.dot( inv(R), inv( inv(R) + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )

def compute_and_c(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return np.dot( inv( inv(R1) + inv(R2) ), inv( inv( inv(R1) + inv(R2) ) + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )

def compute_or_c(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return np.dot( R1 + R2, inv( R1 + R2 + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )

def not_c(C):
    return np.eye( C.shape[0] ) - C

def non_singular_base(C):
    U, s, Uh = linalg.svd(C)
    k = 0
    for singular_val in s:
        if abs(singular_val) > 1e-25:
            k += 1
    return U[:,k:]

def and_c(C1, C2):
    U_sub = non_singular_base(C1)
    V_sub = non_singular_base(C2)
    Base = non_singular_base(U_sub @ U_sub.T + V_sub @ V_sub.T)
    return Base @ inv( Base.T @ (inv(C1) + inv(C2) - np.eye(C1.shape[0])) @ Base) @ Base.T
    #return inv( inv(C1) + inv(C2) - np.eye( C1.shape[0] ) )

def or_c(C1, C2):
    #eye = np.eye(C1.shape[0])
    #return inv( eye + inv( C1 @ inv( eye - C1) + C2 @ inv( eye - C2 ) ) )
    return not_c( and_c( not_c(C1), not_c(C2) ) )

def negative_c(Cs, idx_of_positive_c):
    Cs = Cs[:idx_of_positive_c] + Cs[idx_of_positive_c+1:]
    N = Cs[0]
    for C in Cs[1:]:
        N = or_c( N, C )
    return not_c( N )

#######################################
# run the reservoir with the signal(s) and collect X
Cs = []
X = None
Xtemp = []
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
    Xtemp.append(X_local)
    if X is None:
        X = X_local
        X_delay = X_delay_local
        P = signal[t_washout:]
    else:
        # append new states X_local, X_delay (later used for loading), and pattern
        X = np.concatenate((X, X_local), axis=1)
        X_delay = np.concatenate((X_delay, X_delay_local), axis=1)
        P = np.concatenate((P, signal[t_washout:]))
    Cs.append( compute_c(X_local, aperture) )

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
W = np.dot( np.dot( inv( np.dot( X_delay, X_delay.T ) + reg_W*np.eye(N) ), X_delay ), ( np.arctanh(X)-B ).T ).T
W_out = ( np.dot( np.dot( inv( np.dot( X, X.T ) + reg_out*np.eye(N) ), X), P.T ) ).T

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

times = [ (0,500), (1,500), (2,500) ]
y, X_regen = generate_mixed_signal(times)
ax[3,0].plot(y, label="Generated signal")
ax[4,0].plot(y, label="Generated signal")

#######################################

# Evaluation
def fit(point, Cs, idx):
    # positive evidence
    e_pos = np.dot( np.array(point).T, np.dot(Cs[idx], point) )
    # negative evidence
    e_neg = np.dot( np.array(point).T, np.dot(negative_c(Cs, idx), point) )
    return e_pos + e_neg

def mean_fit(X, Cs, idx):
    sum = 0
    for state_point in zip(*X):
        sum += fit(state_point, Cs, idx) / X.shape[1]
    return sum

#######################################
# K-means

def kmeans(y_length, nb_conceptors):
    # Initial assignments and initial conceptors
    assignments = [ x for x in range(y_length) ]
    new_indices_by_conceptors = []
    np.random.shuffle(assignments)
    for i in range(nb_conceptors):
        new_indices_by_conceptors.append(assignments[i*int(y_length/nb_conceptors):(i+1)*int(y_length/nb_conceptors)])
    # Training loop
    for epoch in range(100):
        print("epoch:",epoch)
        # recompute centroids based on subset of assigned state
        conceptors = [ compute_c(X_regen[:,indices_by_conceptor], aperture) for indices_by_conceptor in new_indices_by_conceptors ]
        # recompute assignments by find the closest conceptor for each of the state points
        old_indices_by_conceptors = new_indices_by_conceptors.copy()
        new_indices_by_conceptors = [ [] for _ in range(nb_conceptors) ]
        for t in range(y_length):
            min_max = 0
            max_conceptor_index = np.random.choice(range(nb_conceptors))
            for conceptor_index in range(len(conceptors)):
                dist = fit(X_regen[:,t],conceptors, conceptor_index) # compute distance from X_regen(:,i) to conceptor
                if dist > min_max:
                    min_max = dist
                    max_conceptor_index = conceptor_index
            new_indices_by_conceptors[ max_conceptor_index ].append(t)

        # stop if converged
        for new_indices_by_conceptor, old_indices_by_conceptor in zip(new_indices_by_conceptors, old_indices_by_conceptors):
            if set(new_indices_by_conceptor) == set(old_indices_by_conceptor):
                print("Converged")
                return conceptors, new_indices_by_conceptors

    return conceptors, new_indices_by_conceptors

conceptors, indices_by_conceptors = kmeans(len(y), 3) # as many conceptors as loaded patterns

# Plot memberships
collection = [ [] for _ in range(len(conceptors)) ]
for t in range(1500):
    for idx, Cl in enumerate(indices_by_conceptors):
        if t in Cl:
            collection[idx].append(1)
        else:
            collection[idx].append(0)

for i, assignments in enumerate(collection):
    ax[3,1].plot(assignments, label="C"+str(i))
#######################################
# Evaluate
kmeans_perf = 0
original_perf = 0

for i in range(len(conceptors)):
    print("#Cl"+str(i),len(indices_by_conceptors[i]))
    perf = mean_fit(X_regen[:,indices_by_conceptors[i]], conceptors, i) / len(y)
    print("perf on cluster ", "Cl"+str(i)+":", perf)
    kmeans_perf += perf

for i, iterations in times:
    i = 0
    original_perf += mean_fit(X_regen[:,i:i+iterations], Cs, i) / len(y)
    i += iterations

# Classify on original conceptors
def plot_classficiation(X, Cs):
    y_len = X.shape[1]
    nb_conceptors = len(Cs)
    indices_by_conceptors = [ [] for _ in range(nb_conceptors) ]
    for t in range(y_len):
        min_dist = INFINITY
        min_conceptor_index = np.random.choice(range(nb_conceptors))
        for conceptor_index in range(len(Cs)):
            dist = fit(X[:,t], Cs, conceptor_index) # compute distance from X_regen(:,i) to conceptor
            if dist < min_dist:
                min_dist = dist
                min_conceptor_index = conceptor_index
        indices_by_conceptors[ min_conceptor_index ].append(t)
    collection = [ [] for _ in range(nb_conceptors) ]
    for t in range(y_len):
        for idx, Cl in enumerate(indices_by_conceptors):
            if t in Cl:
                collection[idx].append(1)
            else:
                collection[idx].append(0)
    for i, assignments in enumerate(collection):
        ax[4,1].plot(assignments, label="C"+str(i))

plot_classficiation(X_regen,Cs)

print("kmeans_perf=",kmeans_perf)
print("original_perf=",original_perf)
for i in range(5):
    ax[i,0].legend()
ax[3,1].legend()
ax[4,1].legend()

plt.show()
fig.suptitle('Aperture='+str(aperture)+', N='+str(N)+', Spec Rad='+str(spectral_radius), fontsize=16)
