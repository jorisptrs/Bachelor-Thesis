import random

from matplotlib import pyplot as plt

random.seed(0)

from lib.conceptors import *
from experiments.exp2_below_phoneme_clustering.kmeans.point import Point
from experiments.exp2_below_phoneme_clustering.kmeans.method import Method



class KMeans:
    """
    This class contains the clustering functionality used by experiment 2.
    It can be used with any of the methods specified in method.py.
    """

    def __init__(self, method=None, Cs=None, signals=None, esn_states=None, XorZ="X", target_sum=None):
        # Constants
        self.method = method
        self.XorZ = XorZ
        self.target_sum = target_sum
        self.points = Point.init_points(signals=signals, esn_states=esn_states, Cs=Cs)
        self.nb_points = len(self.points)

        # Variables
        self.centroids = []

    # ---------------------------------------------------------------------------------------- #
    # Kmeans

    def k_means(self, nb_clusters=4, max_epochs=15, init_clusters="random", save=False, debug=False):
        if debug:
            print(f"Running KMeans: '{init_clusters}' initialization | {self.method} method | {nb_clusters} clusters.")
        ds_hist = []
        centroid_hist = []
        cluster_hist = []

        if init_clusters == "smart":
            self.init_smart(nb_clusters, debug)
        elif init_clusters == "random":
            self.init_random(nb_clusters)
        else:
            # if some initial assignments were passed, e.g., for prior clusters
            self.compute_centroids(init_clusters)

        for epoch in range(max_epochs):
            if debug:
                print("Epoch", epoch, "# centroids:", len(self.centroids))

            if epoch:
                old_clusters = new_clusters.copy()
            new_clusters = [[] for _ in self.centroids]
            distances = []

            # Assignment step
            for point in self.points:
                ds = self.distances_to_centroids(point, normalize=False)
                centroid_index = np.argmin(ds)
                new_clusters[centroid_index].append(point)
                distances.append(np.min(ds))
            # Prevent empty clusters by reassigning
            new_clusters = self.reassign_point_to_empty_clusters(new_clusters)

            # Centroid update step
            self.compute_centroids(new_clusters, rescale=False)

            ds_hist.append(np.mean(distances))
            cluster_hist.append(new_clusters)
            centroid_hist.append(self.centroids)

            # Convergence check
            if epoch and Point.equal_cluster_groups(new_clusters, old_clusters):
                if debug:
                    print("Converged")
                break

        return self.centroids, new_clusters, ds_hist, cluster_hist, centroid_hist

    def mean_intra_cluster_distance(self, clusters):
        ds = []
        self.compute_centroids(clusters)
        for cluster, centroid in zip(clusters, self.centroids):
            for point in cluster:
                ds.append(self.distance_to_centroid(point, centroid))
        return np.mean(ds)

    # ---------------------------------------------------------------------------------------- #
    # Kmeans helpers

    def init_random(self, nb_clusters):
        """
        Assigns points to conceptors according to one of several assignment methods
        Returns [[points in cluster 1], [points in cluster 2], ...]
        """
        clusters = [[] for _ in range(nb_clusters)]
        points = self.points.copy()
        np.random.shuffle(points)
        for i in range(nb_clusters):
            clusters[i] = points[i * int(self.nb_points / nb_clusters): (i + 1) * int(self.nb_points / nb_clusters)]

        self.compute_centroids(clusters)
        return clusters

    def init_smart(self, nb_clusters, debug=False):
        # 1. Find centroids. Adaptation from paper
        points = self.points.copy()
        initial_p = random.choice(points)
        points.remove(initial_p)

        self.centroids = [initial_p]
        self.compute_centroids_Ns()

        for i in range(nb_clusters - 1):
            if debug:
                print("Finding centroid number ", i + 2)
            distances = []

            for point in points:
                ds = self.distances_to_centroids(point)
                distances.append(np.min(ds) ** 2)

            probabilities = distances / np.sum(distances)
            next_p = np.random.choice(points, p=probabilities)
            points.remove(next_p)

            self.centroids.append(next_p)
            self.compute_centroids_Ns()

    def reassign_point_to_empty_clusters(self, new_clusters):
        used_points = []
        while any(len(c) == 0 for c in new_clusters):
            for i, cluster in enumerate(new_clusters):
                if not cluster:
                    print("REASSIGNED POINT TO CLUSTER")
                    farthest_points = []
                    for j, other_cluster in enumerate(new_clusters):
                        if other_cluster:
                            ds = [
                                (self.distance_to_centroid(point, self.centroids[j]) if point not in used_points else 0)
                                for point in other_cluster
                            ]
                            farthest_point = other_cluster[np.argmax(ds)]
                            farthest_points.append((farthest_point, np.max(ds), j))
                    farthest_point_from_own_cluster, _, point_idx = max(farthest_points, key=lambda x: x[1])
                    new_clusters[point_idx].remove(farthest_point_from_own_cluster)
                    new_clusters[i] = [farthest_point_from_own_cluster]
                    used_points.append(farthest_point_from_own_cluster)
                    break
        return new_clusters

    def compute_centroids_Ns(self):
        if not self.method.uses_neg_conceptors():
            return
        Cs = Point.get_Cs(self.centroids)
        Ns = Ns_from_Cs(Cs)
        self.centroids = Point.update_points(self.centroids, Ns=Ns)

    def compute_centroids(self, clusters, method=None, rescale=True):
        if method is None:
            method = self.method

        self.centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                print("GAVE UP CLUSTER ", i, ", SINCE IT HAD NO MEMBERS")
            else:
                mean_signal = np.mean(Point.get_signals(cluster), axis=0)  # For Method.SIGNALS_EUCLIDIAN
                mean_esn_state = np.mean(Point.get_esn_states(cluster), axis=0)  # For Method.PRED_CENTROID
                conceptor = None
                if method.is_in_conceptor_space() or method is Method.STATE_EUCLIDIAN:
                    X = np.array([])
                    for point in cluster:
                        X = np.hstack((X, point.esn_state)) if X.size else point.esn_state
                    if method.is_in_conceptor_space():
                        conceptor = compute_c(X, 1)
                self.centroids.append(Point(signal=mean_signal, C=conceptor, esn_state=mean_esn_state))

        if method.is_in_conceptor_space() and rescale:
            rescaled_Cs = adapt_singular_vals_of_Cs(Point.get_Cs(self.centroids), target_sum=self.target_sum)
            self.centroids = Point.update_points(self.centroids, Cs=rescaled_Cs)
        # for C_kmeans, C_kmeans_recomputed in zip(Cs_kmeans, Cs_kmeans_recomputed):
        # print("Mean divergence: ", d(C_kmeans, C_kmeans_recomputed)/np.linalg.norm(C_kmeans))
        self.compute_centroids_Ns()

    # ---------------------------------------------------------------------------------------- #
    # Distances

    def distance_to_centroid(self, point, centroid, method=None):
        if method is None:
            method = self.method
        dist = None

        if method is Method.CONCEPTOR_SIM:
            dist = 1 - similarity_c(point.C, centroid.C)
        elif method is Method.CONCEPTOR_PRED:
            if centroid.N is None:
                self.compute_centroids_Ns()
            dist = 1 / combined_evidence_vec(point.esn_state, Cs=[centroid.C], idx=0, Ns=[centroid.N])
        elif method is Method.CONCEPTOR_PRED_CS_ONLY:
            dist = 1 / pos_evidence_vec(point.esn_state, Cs=[centroid.C], idx=0)
        elif method is Method.SIGNALS_EUCLIDIAN:
            dist = Point.d(point.signal, centroid.signal)
        elif method is Method.STATE_EUCLIDIAN:
            dist = Point.d(point.esn_state, centroid.esn_state)
        elif method is Method.CONCEPTOR_FROB:
            dist = Point.d(point.C, centroid.C)
        elif method is Method.CONCEPTOR_SPECTRAL:
            dist = spectral_d(point.C, centroid.C)
        elif method is Method.PRED_CENTROIDS:
            dist = (
                           self.distance_to_centroid(point, centroid, Method.CONCEPTOR_PRED) +
                           self.distance_to_centroid(point, centroid, Method.STATE_EUCLIDIAN)
                   ) / 2

        return dist

    def distances_to_centroids(self, point, normalize=True, method=None):
        ds = []

        if method is None:
            method = self.method

        if method is Method.CONCEPTOR_PRED_CS_ONLY:
            es = pos_evidences_for_Cs(point.esn_state, Point.get_Cs(self.centroids),
                                      two_d=self.XorZ == "X")
            ds = [1 / e for e in es]

        elif method is Method.CONCEPTOR_PRED:
            es = evidences_for_Cs(point.esn_state, Point.get_Cs(self.centroids),
                                  Ns=Point.get_Ns(self.centroids), two_d=self.XorZ == "X")
            ds = [1 / e for e in es]

        elif method is Method.PRED_CENTROIDS:
            ds_pred = self.distances_to_centroids(point, normalize=True, method=Method.CONCEPTOR_PRED)
            ds_centroid = self.distances_to_centroids(point, normalize=True, method=Method.STATE_EUCLIDIAN)
            ds = np.add(ds_pred, ds_centroid) / 2

        else:
            ds = [self.distance_to_centroid(point, centroid) for centroid in self.centroids]

        return [d / sum(ds) for d in ds] if normalize else ds

    def mean_dist_to_cluster(self, point, cluster):
        a = 0
        for p2 in cluster:
            if point != p2:
                a += self.distance_to_centroid(point, p2)
        return a / (len(cluster) - 1)

    def assignment_to_cluster_repr(self, assignments):
        points_by_cluster = []
        for cluster in assignments:
            points_by_cluster.append([self.points[index] for index in cluster])
        return points_by_cluster

    def cluster_to_assignment_repr(self, clusters):
        assignments = []
        for cluster in clusters:
            assignments.append([self.points.index(point) for point in cluster])
        return assignments

    def cluster_to_assignment_repr_l(self, cluster_l):
        return [self.cluster_to_assignment_repr(cluster) for cluster in cluster_l]

    @staticmethod
    def remove_empty_clusters(clusters):
        return [c for c in clusters if c]

    def plot_centroid_s(self):
        for i, centroid in enumerate(self.centroids):
            print("N ", i)
            U, s, _ = np.linalg.svd(centroid.C, hermitian=True, full_matrices=False)
            plt.plot(s)
            plt.show()
            print("Sum ", np.sum(s))
            print("Var ", np.var(s))

    def plot_clusters(self, clusters):
        for i, (centroid, cluster) in enumerate(zip(self.centroids, clusters)):
            print("Cluster ", i)
            print("Len: ", len(cluster))
            for p in cluster:
                ds = self.distances_to_centroids(p)
                print(np.argmin(ds))

    def intra_dist_mean(self, clusters):
        sum = 0
        for c, cluster in zip(self.centroids, clusters):
            for p in cluster:
                sum += self.distance_to_centroid(p, c)
        return sum / self.nb_points

    def intra_dist_mean_from_list(self, cluster_l):
        return [self.intra_dist_mean(clusters) for clusters in cluster_l]
