from enum import Enum


class Method(Enum):
    OG_SIGNALS = 1
    CENTROIDS = 2
    SIMS = 3
    PRED = 4
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