import random

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

        if init_clusters == "smart":
            new_clusters = self.assign_to_clusters_smart(nb_clusters, debug)
        elif init_clusters == "random":
            new_clusters = self.assign_to_clusters(nb_clusters)
        else:
            # if some initial assignments were passed, e.g., for prior clusters
            new_clusters = init_clusters

        ds_hist = [self.mean_intra_cluster_distance(new_clusters)]
        cluster_hist = [new_clusters]

        for epoch in range(max_epochs):

            self.compute_centroids(new_clusters)
            if debug:
                print("Epoch", epoch, "# centroids:", len(self.centroids))

            old_clusters = new_clusters.copy()
            new_clusters = [[] for _ in self.centroids]

            distances = []
            for point in self.points:
                # Find closest centroid
                ds = self.distances_to_centroids(point, normalize=False)
                centroid_index = np.argmin(ds)
                new_clusters[centroid_index].append(point)
                distances.append(np.min(ds))
            ds_hist.append(np.mean(distances))
            cluster_hist.append(new_clusters)

            if Point.equal_cluster_groups(new_clusters, old_clusters):
                if debug:
                    print("Converged")
                break

        return self.centroids, new_clusters, ds_hist, cluster_hist

    def mean_intra_cluster_distance(self, clusters):
        ds = []
        self.compute_centroids(clusters)
        for cluster, centroid in zip(clusters, self.centroids):
            for point in cluster:
                ds.append(self.distance_to_centroid(point, centroid))
        return np.mean(ds)

    # ---------------------------------------------------------------------------------------- #
    # Kmeans helpers

    def assign_to_clusters(self, nb_clusters):
        """
        Assigns points to conceptors according to one of several assignment methods
        Returns [[points in cluster 1], [points in cluster 2], ...]
        """
        clusters = [[] for _ in range(nb_clusters)]
        points = self.points.copy()
        np.random.shuffle(points)
        for i in range(nb_clusters):
            clusters[i] = points[i * int(self.nb_points / nb_clusters): (i + 1) * int(self.nb_points / nb_clusters)]
        return clusters

    def assign_to_clusters_smart(self, nb_clusters, debug=False):
        # 1. Find centroids. Adaptation from paper
        clusters = [[] for _ in range(nb_clusters)]
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
                distances.append(np.min(ds))

            probabilities = distances / np.sum(distances)
            next_p = np.random.choice(points, p=probabilities)
            points.remove(next_p)

            self.centroids.append(next_p)
            self.compute_centroids_Ns()

        # 2. Perform assignments
        for point in self.points:
            ds = self.distances_to_centroids(point)
            centroid_index = np.argmin(ds)
            clusters[centroid_index].append(point)

        return clusters

    def compute_centroids_Ns(self):
        if not self.method.uses_neg_conceptors():
            return
        Cs = Point.get_Cs(self.centroids)
        Ns = Ns_from_Cs(Cs)
        self.centroids = Point.update_points(self.centroids, Ns=Ns)

    def compute_centroids(self, clusters, method=None):
        if method is None:
            method = self.method

        self.centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                print("GAVE UP A CLUSTER, SINCE IT HAD NO MEMBERS")
            else:
                mean_signal = np.mean(Point.get_signals(cluster), axis=0) # For Method.OG_SIGNALS
                mean_esn_state = np.mean(Point.get_esn_states(cluster), axis=0)  # For Method.PRED_CENTROID
                conceptor = None
                if method.is_in_conceptor_space() or method is Method.CENTROIDS:
                    X = np.array([])
                    for point in cluster:
                        X = np.hstack((X, point.esn_state)) if X.size else point.esn_state
                    if method.is_in_conceptor_space():
                        conceptor = compute_c(X, 1)
                self.centroids.append(Point(signal=mean_signal, C=conceptor, esn_state=mean_esn_state))

        if method.is_in_conceptor_space():
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

        if method is Method.SIMS:
            dist = 1 - similarity_c(point.C, centroid.C)
        elif method is Method.PRED:
            if centroid.N is None:
                self.compute_centroids_Ns()
            dist = combined_evidence_vec(point.esn_state, Cs=[centroid.C], idx=0, Ns=[centroid.N])
        elif method is Method.OG_SIGNALS:
            dist = Point.d(point.signal, centroid.signal)
        elif method is Method.CENTROIDS:
            dist = Point.d(point.esn_state, centroid.esn_state)
        elif method is Method.PRED_CENTROIDS:
            dist = (
                self.distance_to_centroid(point, centroid, Method.PRED) +
                self.distance_to_centroid(point, centroid, Method.CENTROIDS)
            )/2

        return dist

    def distances_to_centroids(self, point, normalize=True, method=None):
        ds = []

        if method is None:
            method = self.method

        if method is Method.PRED:
            es = evidences_for_Cs(point.esn_state, Point.get_Cs(self.centroids),
                                  Point.get_Ns(self.centroids), two_d=self.XorZ == "X")
            ds = [1/e for e in es]

        elif method.is_in_eucl_space() or method is Method.SIMS:
            ds = [self.distance_to_centroid(point, centroid) for centroid in self.centroids]

        elif method is Method.PRED_CENTROIDS:
            ds_pred = self.distances_to_centroids(point, normalize=True, method=Method.PRED)
            ds_centroid = self.distances_to_centroids(point, normalize=True, method=Method.CENTROIDS)
            ds = np.add(ds_pred, ds_centroid) / 2

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

    @staticmethod
    def remove_empty_clusters(clusters):
        return [c for c in clusters if c]