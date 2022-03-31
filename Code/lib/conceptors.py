from scipy import linalg
import numpy as np

inv = linalg.inv

def corr(X):
    L = X.shape[1] or 1
    return X @ X.T / L

# conceptors
def similarity_c(C1,C2):
    U1, s1, _ = linalg.svd(C1, hermitian=True, full_matrices=False)
    U2, s2, _ = linalg.svd(C2, hermitian=True, full_matrices=False)

    return linalg.norm(
        np.diag(s1)**0.5 @ U1.T @ U2 @ np.diag(s2)**0.5)**2 / (linalg.norm(s1)*linalg.norm(s2))

def compute_c(X, aperture):
    R = corr(X)
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
    return R @ inv( R + aperture ** (-2) * np.eye(R.shape[0]))

def NOT_C(X, aperture):
    R = corr(X)
    return inv(R) @ inv( inv(R) + 1 / np.square(aperture) * np.eye(X.shape[0]) )

def AND(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return ( inv( inv(R1) + inv(R2) ) ) @ inv( inv( inv(R1) + inv(R2) ) + 1 / np.square(aperture) * np.eye(X.shape[0]) )

def OR(X1, X2, aperture):
    R1 = corr(X1)
    R2 = corr(X2)
    return ( R1 + R2 ) @ inv( R1 + R2 + 1 / np.square(aperture) * np.eye(X.shape[0]) )

def NOT_C(C):
    return np.eye( C.shape[0] ) - C

def non_singular_base(C):
    U, s, _ = np.linalg.svd(C, hermitian=True)
    epsilon = 1e-10
    return U[:,np.sum(s>epsilon):]

def AND_C(C1, C2):
    U_sub = non_singular_base(C1)
    V_sub = non_singular_base(C2)
    Base = non_singular_base(U_sub @ U_sub.T + V_sub @ V_sub.T)
    return Base @ inv( Base.T @ (inv(C1) + inv(C2) - np.eye(C1.shape[0])) @ Base) @ Base.T

def OR_C(C1, C2):
    return NOT_C( AND_C( NOT_C(C1), NOT_C(C2) ) )

def negative_c(Cs, idx_of_positive_c):
    Cs = Cs[:idx_of_positive_c] + Cs[idx_of_positive_c+1:]
    N = Cs[0]
    for C in Cs[1:]:
        N = OR_C( N, C )
    return NOT_C( N )

# Evaluation
def fit(point, Cs, idx):
    """
    Returns combined evidence that a state point corresponds to the conceptor Cs[idx]
    """
    # positive evidence
    e_pos = np.array(point).T @ Cs[idx] @ point
    # negative evidence
    e_neg = np.array(point).T @ negative_c(Cs, idx) @ point
    return e_pos + e_neg

def mean_fit(X, Cs, idx):
    sum = 0
    for state_point in zip(*X):
        sum += fit(state_point, Cs, idx) / X.shape[1]
    return sum


def find_closest_C(x, Cs):
    """
    Returns the index of the conceptor whose combined evidence for x is highest
    """
    max_dist = 0
    max_conceptor_index = np.random.choice(range(len(Cs)))
    for conceptor_index in range(len(Cs)):
        dist = fit(x, Cs, conceptor_index) # compute distance from X_regen(:,i) to conceptor
        if dist > max_dist:
            max_dist = dist
            max_conceptor_index = conceptor_index
    return max_conceptor_index, max_dist

def test(X, Cs, mode="PROP"):
    """
    For each conceptor, collects 1 when it closest models the time-step, otherwise 0
    """
    y_len = X.shape[1]
    collection = [ [] for _ in range(len(Cs)) ]
    mean_dist = 0
    for t in range(y_len):
        if mode == "WGA": # Winner gets all
            closest_C, dist = find_closest_C(X[:,t], Cs)
            mean_dist += dist / y_len
            for c_idx in range(len(Cs)):
                if c_idx == closest_C:
                    collection[ c_idx ].append(1)
                else:
                    collection[ c_idx ].append(0)
        elif mode == "PROP": # proportional
            sum_dist = 0
            for c_idx in range(len(Cs)):
                dist = fit(X[:,t], Cs, c_idx)
                sum_dist += dist
                collection[ c_idx ].append(dist)
                mean_dist += dist / y_len
    return collection, mean_dist