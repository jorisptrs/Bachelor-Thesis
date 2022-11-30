"""
Custom Similarity/Distance function for conceptors
"""

from scipy import linalg
import numpy as np


def similarity_c_custom(C1,C2,dist_fun):
    U1, s1, _ = np.linalg.svd(C1, hermitian=True, full_matrices=False)
    U2, s2, _ = np.linalg.svd(C2, hermitian=True, full_matrices=False)

    if dist_fun == "eucl":
        d = 0

        U1 = U1 @ np.diag(s1)
        U2 = U2 @ np.diag(s2)
        for i1, row1 in enumerate(U1):
            for i2, row2 in enumerate(U2):
                aligned_row2 = (1 - 2 * (row1 @ row2 < 0)) * row2
                d += linalg.norm(row1 - aligned_row2)

        return d / (linalg.norm(s1)+linalg.norm(s2))

    elif dist_fun == "cos":
        return linalg.norm(
            np.diag(s1)**0.5 @ U1.T @ U2 @ np.diag(s2)**0.5)**2 / (linalg.norm(s1)*linalg.norm(s2)
        )

# dist_fun = 'eucl'
# d = similarity_c_custom(Cs[phonemes.index("f")],Cs[phonemes.index("h#")],dist_fun)
# print(d)