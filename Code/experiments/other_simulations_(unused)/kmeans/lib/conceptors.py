import numpy as np
from scipy import linalg

inv = linalg.inv


def corr(X, weights=None):
    if weights:
        norm = sum(weights) or 1
        X = X * np.sqrt(np.array(weights))
    else:
        norm = X.shape[1] or 1  # L
    return X @ X.T / norm


def NRMSE(signal, signal_truth, max_shift):
    """
    Compute normalized root mean squared error between a (trained) signal and
    and ground truth signal. Attempts to correct for phase shifts below max_shift.
    """
    K = len(signal) - max_shift

    mse = [
        np.mean([(signal[(i + d)] - signal_truth[i]) ** 2 for i in range(K)])
        for d in range(max_shift)
    ]

    nrmse = np.sqrt(min(mse) / np.mean(signal_truth ** 2))
    return nrmse


def walking_NRMSE(signal, signal_truth, window, max_shift):
    """
    Allows a shifts for each window time steps throughout the signals
    """
    NRMSEs = []
    for t in range(len(signal)):
        if t % window == 0:
            NRMSEs.append(NRMSE(signal[t:t + window], signal_truth[t:t + window], max_shift))
    return np.mean(np.array(NRMSEs))


# conceptors
def similarity_c(C1, C2):
    U1, s1, _ = np.linalg.svd(C1, hermitian=True, full_matrices=False)
    U2, s2, _ = np.linalg.svd(C2, hermitian=True, full_matrices=False)

    return linalg.norm(
        np.diag(s1) ** 0.5 @ U1.T @ U2 @ np.diag(s2) ** 0.5) ** 2 / (linalg.norm(s1) * linalg.norm(s2))


def max_similarity(Cs, Cs_truth):
    sum = 0
    for C in Cs:
        sum += max([similarity_c(C, C_truth) for C_truth in Cs_truth])
    return sum


def compute_c(X, aperture, weights=None):
    """
    Computes conceptor. Potentially using via a weighted correlation matrix.
    param :Weights: weights for each time step (column) in X
    """
    R = corr(X, weights)
    # Eig_vals, Eig_vecs = np.linalg.eig(R)
    # U = Eig_vecs
    # Sigma = np.diag(Eig_vals)
    # for elem in Eig_vals:
    #     if abs(elem) < 1e-100:
    #         print("!!! Zero singular value(s) !!!")
    # print(X.shape)
    # print(R.shape)
    # print(np.min(R))
    # print(np.max(R))
    return R @ inv(R + aperture ** (-2) * np.eye(R.shape[0]))


def normalize_apertures(Cs):
    """
    Normalize all conceptors to the average of their summed singular values
    """
    sum_avg = 0
    normalized_Cs = []
    for C in Cs:
        s = np.diag(C)
        # _, s, _ = np.linalg.svd(C, hermitian=True)
        sum_avg += np.sum(s) / len(Cs)
    for C in Cs:
        # _, s, _ = np.linalg.svd(C, hermitian=True)
        s = np.diag(C)
        print(C.shape)
        print(C)
        sum_local = np.sum(s)
        normalized_Cs.append(C * sum_avg / sum_local)
        print(sum_local)
    print(sum_avg)
    return normalized_Cs


def NOT_C(X, aperture):
    R = corr(X)
    return inv(R) @ inv(inv(R) + 1 / np.square(aperture) * np.eye(X.shape[0]))


def AND(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return (inv(inv(R1) + inv(R2))) @ inv(inv(inv(R1) + inv(R2)) + 1 / np.square(aperture) * np.eye(X.shape[0]))


def OR(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return (R1 + R2) @ inv(R1 + R2 + 1 / np.square(aperture) * np.eye(X.shape[0]))


def NOT_C(C):
    return np.eye(C.shape[0]) - C


def non_singular_base(C):
    U, s, _ = np.linalg.svd(C, hermitian=True)
    epsilon = 1e-10
    return U[:, np.sum(s > epsilon):]


def AND_C(C1, C2):
    U_sub = non_singular_base(C1)
    V_sub = non_singular_base(C2)
    Base = non_singular_base(U_sub @ U_sub.T + V_sub @ V_sub.T)
    return Base @ inv(Base.T @ (inv(C1) + inv(C2) - np.eye(C1.shape[0])) @ Base) @ Base.T


def OR_C(C1, C2):
    return NOT_C(AND_C(NOT_C(C1), NOT_C(C2)))


def negative_c(Cs, idx_of_positive_c):
    Cs = Cs[:idx_of_positive_c] + Cs[idx_of_positive_c + 1:]
    N = Cs[0]
    for C in Cs[1:]:
        N = OR_C(N, C)
    return NOT_C(N)


def Ns_from_Cs(Cs):
    return [negative_c(Cs, idx) for idx in range(len(Cs))]


# Evaluation
def combined_evidence(point, Cs, idx):
    """
    Returns combined evidence that a state point corresponds to the conceptor ps[idx]
    """
    # positive evidence
    e_pos = np.array(point).T @ Cs[idx] @ point
    # negative evidence
    e_neg = np.array(point).T @ negative_c(Cs, idx) @ point
    return e_pos + e_neg


def combined_evidence_vec(X, Cs, idx, Ns=None):
    """
    Vectorized combined evidences
    """
    # positive evidence
    e_pos = X.T @ Cs[idx] @ X
    # negative evidence
    if Ns:
        e_neg = X.T @ Ns[idx] @ X
    else:
        e_neg = X.T @ negative_c(Cs, idx) @ X
    return e_pos + e_neg


def mean_combined_evidence(X, Cs, assignments):
    sum = 0
    for idx, assigned_time_steps in enumerate(assignments):
        for t in assigned_time_steps:
            sum += combined_evidence(X[:, t], Cs, idx) / X.shape[1]
    return sum


def weighted_mean_combined_evidence(X, Cs, assignments):
    sum = 0
    nb_points = X.shape[1]
    for idx, assignment in enumerate(assignments):
        for t in range(nb_points):
            sum += assignment[t] * combined_evidence(X[:, t], Cs, idx)
    return sum


def normalized_evidences_by_conceptor(x, Cs):
    """
    Return evidences for each conceptor normalized to sum to 1
    """
    evidences = []
    for conceptor_index in range(len(Cs)):
        evidences.append(combined_evidence(x, Cs, conceptor_index))
    return np.array(evidences) / sum(evidences)


def find_closest_C(x, Cs):
    """
    Returns the index of the conceptor whose combined evidence for x is highest
    """
    max_dist = 0
    max_conceptor_index = np.random.choice(range(len(Cs)))
    for conceptor_index in range(len(Cs)):
        evidence = combined_evidence(x, Cs, conceptor_index)  # compute distance from X_regen(:,i) to conceptor
        if evidence > max_dist:
            max_dist = evidence
            max_conceptor_index = conceptor_index
    return max_conceptor_index, max_dist


def test(X, Cs, mode="PROP"):
    """
    For each conceptor, collects 1 when it closest models the time-step, otherwise 0
    """
    y_len = X.shape[1]
    collection = [[] for _ in range(len(Cs))]
    mean_dist = 0
    for t in range(y_len):
        if mode == "WGA":  # Winner gets all
            closest_C, dist = find_closest_C(X[:, t], Cs)
            mean_dist += dist / y_len
            for c_idx in range(len(Cs)):
                if c_idx == closest_C:
                    collection[c_idx].append(1)
                else:
                    collection[c_idx].append(0)
        elif mode == "PROP":  # proportional
            sum_dist = 0
            for c_idx in range(len(Cs)):
                dist = combined_evidence(X[:, t], Cs, c_idx)
                sum_dist += dist
                collection[c_idx].append(dist)
                mean_dist += dist / y_len
    return collection, mean_dist
