import math

import numpy as np

from experiments.exp2_below_phoneme_clustering.kmeans.kmeans import KMeans


def I(assignments, correct_assignments):
    I_res = 0
    n_samples = sum([len(x) for x in assignments])
    for assignment in assignments:
        for correct_assignment in correct_assignments:
            p_intersec = len([elem for elem in assignment if elem in correct_assignment]) / n_samples
            p_assignment = len(assignment) / n_samples
            p_correct_assignment = len(correct_assignment) / n_samples
            if p_intersec != 0 and p_assignment != 0 and p_correct_assignment != 0:
                I_res += p_intersec * np.log(p_intersec / (p_assignment * p_correct_assignment))
    return I_res


def H(clusters):
    H = 0
    n_samples = sum([len(x) for x in clusters])
    for cluster in clusters:
        p_cluster = len(cluster) / n_samples
        if len(cluster):
            H -= p_cluster * np.log(p_cluster)
    return H


def NMI(assignments, correct_assignments):
    Is = I(assignments, correct_assignments)
    Ha = H(assignments)
    Hc = H(correct_assignments)
    return 2 * Is / (Ha + Hc)


def NMIs_from_list(assignment_list, correct_assignments):
    return [ NMI(a, correct_assignments) for a in assignment_list ]


def best_match(assignments, correct_assignments):
    max_matches = 0
    overlaps = np.zeros((len(correct_assignments), 1))
    for j, assignment in enumerate(assignments):
        for i, correct_assignment in enumerate(correct_assignments):
            if assignment in correct_assignment:
                overlaps[i] += 1
    return np.argmax(overlaps), np.max(overlaps)

def TP(assignments, correct_assignments):
    tp = 1
    for assignment in assignments:
        Q = best_match(assignment, correct_assignments)[1]
        tp += math.comb(Q, 2)


def TPFP(assignments, correct_assignments):
    TPFP = 0
    for assignment in assignments:
        TPFP += math.comb(len(assignment), 2)


def TPFN(assignments, correct_assignments):
    TPFN = 0
    for correct_assignment in correct_assignments:
        TPFN += math.comb(len(assignment), 2)
    return TPFN


def prc(assignments, correct_assignments):
    return TP(assignments, correct_assignments) / TPFP(assignments, correct_assignments)


def rec(assignments, correct_assignments):
    return TP(assignments, correct_assignments) / TPFN(assignments, correct_assignments)


def F(assignments, correct_assignments):
    prc = prc(assignments, correct_assignments)
    rec = rec(assignments, correct_assignments)
    return 2 * prc * rec / (prc * rec)


def F(assignments, correct_assignments):
    overlaps = np.zeros((len(assignments), len(correct_assignments)))  # (ps x phonemes)

    for i1, assignment in enumerate(assignments):
        for a in assignment:
            for i2, correct_assignment in enumerate(correct_assignments):
                if a in correct_assignment:
                    overlaps[i1, i2] += 1

    dic = []
    cnt = 0
    for _ in range(len(correct_assignments)):
        i1, i2 = np.unravel_index(overlaps.argmax(), overlaps.shape)
        dic.append(i1)
        cnt += overlaps[i1, i2]
        overlaps[i1, :] = np.zeros((1, len(correct_assignments)))
        overlaps[:, i2] = np.zeros((len(assignments),))

    return cnt / max([max(ass) for ass in assignments if ass != []])


class Silhouette:
    def __init__(self, km):
        self.km = km

    def silh_aux(self, point, clusters):
        a = 0
        b = 0
        for i, cluster in enumerate(clusters):
            dist = self.km.distance_to_centroid(point, centroid=self.km.centroids[i])
            if point in cluster:
                if len(cluster) == 1:
                    return 0
                a = dist
            else:
                b = dist if not b else min(b, dist)
        return (b - a) / max(a, b)

    def simpl_silh(self, clusters):
        clusters = self.km.remove_empty_clusters(clusters)
        self.km.compute_centroids(clusters)
        return np.mean([self.silh_aux(point, clusters=clusters) for point in self.km.points])

    def simpl_silh_from_list(self, cluster_hist):
        return [ self.simpl_silh(cluster) for cluster in cluster_hist ]

def get_heat_map(ps, sim_func, zero_diag=True):
    heat_map = np.zeros((len(ps), len(ps)))

    for x in range(len(ps)):
        for y in range(0, x+(not zero_diag)):
            sim = sim_func(ps[x], ps[y])
            heat_map[x, y] = sim
            heat_map[y, x] = sim

    return heat_map