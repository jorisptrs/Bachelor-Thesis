import numpy as np


class Point:
    def __init__(self, signal=None, esn_state=None, C=None, N=None):
        self.signal = signal
        self.esn_state = esn_state
        self.C = C
        self.N = N

    def __eq__(self, other):
        return np.array_equal(self.signal, other.signal) \
               and np.array_equal(self.esn_state, other.esn_state) \
               and np.array_equal(self.C, other.C) \
               and np.array_equal(self.N, other.N)

    @staticmethod
    def equal_clusters(cluster1, cluster2):
        for p1 in cluster1:
            found_match = False
            for p2 in cluster2:
                if p1 == p2:
                    found_match = True
            if not found_match:
                return False
        return True

    @staticmethod
    def equal_cluster_groups(clusters1, clusters2):
        for cluster1 in clusters1:
            found_match = False
            for cluster2 in clusters2:
                if Point.equal_clusters(cluster1, cluster2):
                    found_match = True
            if not found_match:
                return False
        return True

    @staticmethod
    def init_points(signals=None, esn_states=None, Cs=None):
        points = []
        for i in range(2 ** 16):
            new_point = Point()
            if signals is not None:
                if i > len(signals) - 1:
                    return points
                new_point.signal = signals[i]
            if esn_states is not None:
                if i > len(esn_states) - 1:
                    return points
                new_point.esn_state = esn_states[i]
            if Cs is not None:
                if i > len(Cs) - 1:
                    return points
                new_point.C = Cs[i]
            points.append(new_point)
        return points

    @staticmethod
    def get_Cs(l):
        return [c.C for c in l]

    @staticmethod
    def get_Ns(l):
        return [c.N for c in l]

    @staticmethod
    def get_signals(l):
        return [c.signal for c in l]

    @staticmethod
    def get_esn_states(l):
        return [c.esn_state for c in l]

    @staticmethod
    def update_points(l, Cs=None, Ns=None):
        updated = None
        if Cs is not None:
            updated = []
            for p, C in zip(l, Cs):
                p.C = C
                updated.append(p)
            l = updated
        if Ns is not None:
            updated = []
            for p, N in zip(l, Ns):
                p.N = N
                updated.append(p)
        return updated

    # Euclidian distance / Frobenius norm
    @staticmethod
    def d(x, y):
        return np.linalg.norm(x - y)
