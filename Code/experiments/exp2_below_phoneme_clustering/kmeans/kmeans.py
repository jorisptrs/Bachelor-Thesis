import random

random.seed(0)

from lib.conceptors import *
from point import Point
from method import Method

class KMeans:

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

    def k_means(self, nb_clusters=4, max_epochs=15, init_clusters="random", save=False):
        print(f"Running KMeans: '{init_clusters}' initialization | {self.method} method.")

        if init_clusters == "smart":
            new_clusters = self.assign_to_clusters_smart(nb_clusters)
        elif init_clusters == "random":
            new_clusters = self.assign_to_clusters(nb_clusters)
        else:
            # if some initial assignments were passed, e.g., for prior clusters
            new_clusters = init_clusters

        for epoch in range(max_epochs):

            self.compute_centroids(new_clusters)
            print("Epoch ", epoch, "# centroids:", len(self.centroids))

            old_clusters = new_clusters.copy()
            new_clusters = [[] for _ in self.centroids]

            for point in self.points:
                # Find closest centroid
                ds = self.distances_to_centroids(point, normalize=False)
                centroid_index = np.argmin(ds)
                new_clusters[centroid_index].append(point)

            if Point.equal_cluster_groups(new_clusters, old_clusters):
                print("Converged")
                break

        return self.centroids, new_clusters

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

    def assign_to_clusters_smart(self, nb_clusters):
        # 1. Find centroids. Adaptation from paper
        clusters = [[] for _ in range(nb_clusters)]
        points = self.points.copy()
        initial_p = random.choice(points)
        points.remove(initial_p)

        self.centroids = []
        self.centroids.append(points[initial_p])
        self.compute_centroids_Ns()

        for i in range(nb_clusters - 1):
            print("Finding centroid number ", i + 2)
            probabilities = []

            for point in points:
                ds = self.distances_to_centroids(point)
                probabilities.append(np.min(ds))

            next_p = np.random.choice(points, p=probabilities / np.sum(probabilities))
            points.remove(next_p)

            self.centroids.append(next_p)
            self.compute_centroids_Ns()

        # 2. Perform assignments
        for point in self.points:
            ds = self.distances_to_centroids(point)
            centroid_index = np.min(ds)
            clusters[centroid_index].append(point)

        return clusters

    def compute_centroids_Ns(self):
        if not self.method.uses_neg_conceptors():
            return
        Cs = Point.get_Cs(self.centroids)
        Ns = Ns_from_Cs(Cs)
        self.centroids = Point.update_points(self.centroids, Ns=Ns)

    def compute_centroids(self, clusters):
        self.centroids = []
        if self.method.is_in_conceptor_space():
            centroids = []
            for cluster in clusters:
                X = np.array([])
                for point in cluster:
                    X = np.hstack((X, point.esn_state)) if X.size else point.esn_state
                if X.size:
                    centroids.append(Point(C=compute_c(X, 1)))
            rescaled_Cs = adapt_singular_vals_of_Cs(Point.get_Cs(centroids), target_sum=self.target_sum)
            self.centroids = Point.update_points(centroids, Cs=rescaled_Cs)
            # for C_kmeans, C_kmeans_recomputed in zip(Cs_kmeans, Cs_kmeans_recomputed):
            # print("Mean divergence: ", d(C_kmeans, C_kmeans_recomputed)/np.linalg.norm(C_kmeans))
            self.compute_centroids_Ns()
        elif self.method is Method.OG_SIGNALS:
            for cluster in clusters:
                if cluster:
                    new_centroid = np.mean(Point.get_signals(cluster), axis=0)
                    self.centroids.append(Point(signal=new_centroid))
        elif self.method is Method.CENTROIDS:
            for cluster in clusters:
                if cluster:
                    new_centroid = np.mean(Point.get_esn_states(cluster), axis=0)
                    self.centroids.append(Point(esn_state=new_centroid))

    # ---------------------------------------------------------------------------------------- #
    # Distances

    def distance_to_centroid(self, point, centroid):
        dist = None

        if self.method is Method.SIMS:
            dist = 1 - similarity_c(point.C, centroid.C)
        elif self.method is Method.PRED:
            if centroid.N is None:
                self.compute_centroids_Ns()
            dist = combined_evidence_vec(point.signal, [centroid.C], centroid.N)
        elif self.method is Method.OG_SIGNALS:
            dist = Point.d(point.signal, centroid.signal)
        elif self.method is Method.CENTROIDS:
            dist = Point.d(point.esn_state, centroid.esn_state)

        return dist

    def distances_to_centroids(self, point, normalize=True, method=None):
        ds = []

        if method is None:
            method = self.method

        if method is Method.PRED:
            es = evidences_for_Cs(point.esn_state, Point.get_Cs(self.centroids),
                                  Point.get_Ns(self.centroids), two_d=self.XorZ == "X")
            ds = [max(es) - e for e in es]

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
