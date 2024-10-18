from enum import Enum


class Method(Enum):
    """
    The possible methods for clustering. They affect the centroid computation and the cluster assignment steps.
    """

    SIGNALS_EUCLIDEAN = 1 # Mean signals centroids & euclidean distance function (applied pairwise to corresponding timesteps)
    STATE_EUCLIDEAN = 2 # Mean reservoir state centroids & euclidean distance function (applied pairwise to corresponding timesteps)
    CONCEPTOR_SIM = 3 # Conceptor centroids & conceptor similarity-based distance function
    CONCEPTOR_PRED = 4 # Conceptor centroids & evidence distance function
    CONCEPTOR_PRED_CS_ONLY = 5
    PRED_CENTROIDS = 6
    CONCEPTOR_FROB = 7
    CONCEPTOR_SPECTRAL = 8

    def is_in_conceptor_space(self):
        return self in [Method.CONCEPTOR_PRED, Method.CONCEPTOR_SIM, Method.PRED_CENTROIDS,
                        Method.CONCEPTOR_FROB, Method.CONCEPTOR_SPECTRAL, Method.CONCEPTOR_PRED_CS_ONLY]

    def is_in_eucl_space(self):
        return self in [Method.SIGNALS_EUCLIDEAN, Method.STATE_EUCLIDEAN]

    def uses_neg_conceptors(self):
        return self in [Method.CONCEPTOR_PRED, Method.PRED_CENTROIDS]

    @staticmethod
    def get_all():
        return [m.name for m in Method]
