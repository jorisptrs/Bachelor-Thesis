import numpy as np
from scipy import linalg, sparse

inv = linalg.inv


class ESN:
    def __init__(self, esn_params):
        self.in_dim = esn_params["in_dim"]  # W_in (N x (1 + in_dim))
        self.out_dim = esn_params["out_dim"]  # W_out (out_dim x N)
        self.N = esn_params["N"]  # reservoir size
        self.W_in = esn_params["W_in_scale"] * np.random.normal(0, 1, (self.N, self.in_dim))
        self.b = esn_params["b_scale"] * np.random.normal(0, 1, (self.N, 1))
        self.W = sparse.random(self.N, self.N, density=.1).toarray()  # W (N x N)
        self.W_out = None
        self.spectral_radius = esn_params["spectral_radius"]
        self.set_spectral_radius(self.spectral_radius)

    def set_spectral_radius(self, radius):
        """
        Set the spectral radius
        :param radius: desired spectral radius
        """
        spectral_radius_old = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= radius / spectral_radius_old

    def generate(self, Cs, assignments, y_length, fuzzy=False):
        """
        List of conceptor
        Map from conceptor index to number of timesteps
        """
        x = np.random.normal(0, 1, (self.N, 1))
        X_regen = np.zeros((self.N, y_length))  # collection matrix
        y = np.zeros(y_length)
        # Generate signal
        for t in range(y_length):
            if fuzzy:
                C_lin_comb = np.zeros((self.N, self.N))
                for i, C in enumerate(Cs):
                    C_lin_comb += assignments[i][t] * C
                x = C_lin_comb @ np.tanh(np.reshape(self.b, (self.N, 1)) + (self.W @ x))
            else:
                C_idx = 0
                for idx, iterations in enumerate(assignments):
                    if t in iterations:
                        C_idx = idx
                x = Cs[C_idx] @ np.tanh(np.reshape(self.b, (self.N, 1)) + (self.W @ x))
            X_regen[:, t] = x[:, 0]
            y[t] = self.W_out @ x
        return y, X_regen

    def run(self, signal, t_washout):
        """
        p_idx (in_dim x t_max)
        """
        t_max = signal.shape[0]
        L = max(0, t_max - t_washout)  # steps after washout

        X = np.zeros((self.N, L))
        X_delay = np.zeros((self.N, L))

        x = np.random.normal(0, 1, (self.N, 1))

        for t in range(t_max):
            if (t >= t_washout):
                X_delay[:, t - t_washout] = x[:, 0]
            p = signal[t]
            x = np.tanh(np.dot(self.W, x) + np.dot(self.W_in, p) + self.b)
            if (t >= t_washout):
                X[:, t - t_washout] = x[:, 0]
        return X, X_delay

    def load(self, X, X_delay, reg_W):
        """
        load reservoir by updating W and W_out by ridge regression
        :param reg_W: regularization coefficient for internal weights
        """
        B = np.tile(self.b, X.shape[1])
        self.W = (
                inv((X_delay @ X_delay.T) + reg_W * np.eye(self.N))
                @ X_delay
                @ (np.arctanh(X) - B).T
        ).T

    def train_out_identity(self, X, P, reg_out):
        """
        :param reg_out: regularization coefficient for output weights
        """
        self.W_out = (inv((X @ X.T) + reg_out * np.eye(self.N)) @ X @ P.T).T
