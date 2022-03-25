from cmath import inf
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg, sparse, interpolate

# numpy.linalg is also an option for even fewer dependencies

fig, ax = plt.subplots(2, 4)

# np.random.seed(0)
t_max = 1500
t_test = 50
t_washout = 500 # number of washout steps
L = max(0, t_max - t_washout) # steps after washout
aperture = 5
in_dim = out_dim = 1
N = 100 # reservoir size

# collect data
def gen_signal(n, period, amplitude):
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1/period) * ts)
    return data

p1 = gen_signal(t_max, 5, 1)
p2 = gen_signal(t_max, 10, 1)

data = [p1, p2]

# init reservoir
W_in = 1.5 * np.random.normal(0, 1, (N,in_dim))
b = 0.2 * np.random.normal(0, 1, (N,1))
W_star = sparse.random(N, N, density=.1).toarray()
W_out = None

# set the spectral radius
spectral_radius_old = np.max(np.abs(np.linalg.eigvals(W_star)))
spectral_radius = 1.5
W_star *= spectral_radius / spectral_radius_old

# conceptor ingredients
def compute_conceptor(X, aperture):
    R = np.dot( X, X.T )/X.shape[1]
    Eig_vals, Eig_vecs = np.linalg.eig(R)
    U = Eig_vecs
    Sigma = np.diag(Eig_vals)
    # plot log singular values
    ax[i,2].plot( np.log(Eig_vals) )
    ax[i,3].plot( Eig_vals[:10] )
    return np.dot( R, np.linalg.inv( R + 1 / np.square(aperture) * np.eye(X.shape[0]) ) )
    
ax[0,2].title.set_text("Log Energy (singular values)")
ax[0,3].title.set_text("Top Energy (singular values)")

# run the reservoir with the signal(s) and collect X
Cs = []
X = None
X_delay = None
P = None
for i, signal in enumerate(data):
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
        X = np.concatenate((X, X_local), axis=1)
        X_delay = np.concatenate((X_delay, X_delay_local), axis=1)
        P = np.concatenate((P, signal[t_washout:]))
    Cs.append( compute_conceptor(X_local, aperture) )

    # Plot some activity
    ax[i,1].plot( X_local[0,:t_test] )
    ax[i,1].plot( X_local[9,:t_test] )
    ax[i,1].plot( X_local[59,:t_test] )
    ax[i,1].plot( X_local[99,:t_test] )

ax[0,1].title.set_text("Example neuron values")

# train W and W_out by ridge regression
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

# Testing
for i, signal in enumerate(data):
    x = np.random.normal(0, 1, (N,1))
    y = np.zeros(t_test)

    # Regenerate signal
    for t in range(t_test):
        x = np.dot( Cs[i], np.tanh( np.reshape(b,(N,1)) + np.dot( W, x ) ) )
        y[t] = np.dot( W_out, x )
    
    min_rnmse = inf
    for phase in range(15):
        #y_cs = interpolate.CubicSpline(np.arange(len(shift(y,phase))), y)
        #x_sample = np.arange(t_test*20)
        #y_sample = y_cs(x_sample)
        x_sample = np.arange(t_test)
        y_sample = y
        # compute MSE over test set
        nrmse = np.sqrt( np.mean( np.square( signal[t_washout+1:t_washout+t_test+1] - y_sample[0:t_test] ) ) / np.mean( np.square( y_sample[0:t_test] ) ) )
        if (nrmse < min_rnmse):
            min_rnmse = nrmse
            y = shift(y,phase)

    print('NRMSE = ' + str( min_rnmse))

    ax[i,0].plot(signal[:t_test], label="Original signal")
    ax[i,0].plot(y, label="Regenerated signal")
    
    ax[i,0].annotate('NRMSE='+str(round(nrmse,2)), xy=(0, 0), xytext=(5, -80), textcoords='offset points',
             size=13, bbox=dict(boxstyle ="round", fc ="0.8"))
    ax[i,0].legend(loc="upper left")

ax[0,0].title.set_text("Comparison of signals")
fig.suptitle('Aperture='+str(aperture)+', N='+str(N)+', Spec Rad='+str(spectral_radius)+', other params as in Herbert (2014)', fontsize=16)
plt.show()
