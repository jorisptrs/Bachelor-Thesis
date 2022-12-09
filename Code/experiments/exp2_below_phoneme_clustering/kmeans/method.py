from enum import Enum


class Method(Enum):
    """
    The possible methods for clustering. They affect the centroid computation and the cluster assignment steps.
    """

    OG_SIGNALS = 1 # Mean signals centroids & euclidian distance function (applied pairwise to corresponding timesteps)
    CENTROIDS = 2 # Mean reservoir state centroids & euclidian distance function (applied pairwise to corresponding timesteps) (TODO: give a more intuitive name)
    SIMS = 3 # Conceptor centroids & conceptor similarity-based distance function
    PRED = 4 # Conceptor centroids & evidence distance function
    PRED_CENTROIDS = 5

    def is_in_conceptor_space(self):
        return self in [Method.PRED, Method.SIMS, Method.PRED_CENTROIDS]

    def is_in_eucl_space(self):
        return self in [Method.OG_SIGNALS, Method.CENTROIDS]

    def uses_neg_conceptors(self):
        return self in [Method.PRED, Method.PRED_CENTROIDS]

    @staticmethod
    def get_all():
        return [m.name for m in Method]
