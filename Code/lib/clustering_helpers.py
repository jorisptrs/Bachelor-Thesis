import math

import numpy as np

from lib.conceptors import Ns_from_Cs, similarity_c


def d(x, y):
    return np.linalg.norm(x - y)


def mean_dist_to_cluster(x1_idx, Cs, assignment):
    a = 0
    for x2_idx in assignment:
        if x != x2:
            a += distance(Cs[x1_idx], Cs[x2_idx])
    return a / (len(Cs) - 1)


def silh_aux(x, Cs, assignments):
    a = 0
    b = 0
    for assignment in assignments:
        if x in assingment:
            if len(assignment) == 1:
                return 0
            a = mean_dist_to_cluster(x, Cs, assignment)
        else:
            b = min(b, mean_dist_to_cluster(x, Cs, assignment))
    return (b - a) / max(a, b)


def dist_matrix(Cs):
    heat_map = np.zeros((len(Cs), len(Cs)))

    for x in range(len(Cs)):
        for y in range(0, x + 1):
            sim = similarity_c(Cs[x], Cs[y])
            heat_map[x, y] = sim
            heat_map[y, x] = sim


def silh(Cs, assignments):
    return mean([silh_aux(x, Cs, assingments) for x in range(len(Cs))])


def assign_to_clusters(nb_points, nb_clusters):
    """
    Assigns points to conceptors according to one of several assignment methods
    Returns [[points in cluster 1], [points in cluster 2], ...]
    """
    np.random.seed(2)
    assignments = [[] for _ in range(nb_clusters)]
    points = [x for x in range(nb_points)]
    np.random.shuffle(points)
    for i in range(nb_clusters):
        assignments[i] = points[i * int(nb_points / nb_clusters):(i + 1) * int(nb_points / nb_clusters)]
    return assignments


def assign_to_clusters_smart(method, features, Cs, nb_clusters):
    # 1. Find centroids. Adaptation from paper
    assignments = [[] for _ in range(nb_clusters)]
    centroids = []
    points = list(range(len(features)))
    initial_p = random.choice(points)
    points.remove(initial_p)
    if method == "pred" or method == "sims":
        centroids.append(Cs[initial_p])
    else:
        centroids.append(features[initial_p])

    for i in range(nb_clusters - 1):
        print("Finding centroid number ", i + 2)
        probabilities = []

        Ns = Ns_from_Cs(centroids) if method == "pred" else None

        for point in points:
            ds = find_distances_to_centroids(method, point, features[point], Cs, centroids, Ns)
            probabilities.append(np.min(ds))

        next_p = np.random.choice(points, p=probabilities / np.sum(probabilities))
        points.remove(next_p)

        if method == "pred" or method == "sims":
            centroids.append(Cs[next_p])
        else:
            centroids.append(features[next_p])

    Ns = Ns_from_Cs(centroids) if method == "pred" else None

    # 2. Perform assignments
    for p, feature in enumerate(features):
        centroid_index = find_closest_centroid(method, p, feature, Cs, centroids, Ns)
        assignments[centroid_index].append(p)
    return assignments


def I(assignments, correct_assignments):
    I_res = 0
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


def best_match(assignment, correct_assignments):
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


def prc(assignments, correct_assignments):
    return TP(assignments, correct_assignments) / TPFP(assignments, correct_assignments)


def rec(assignments, correct_assignments):
    return TP(assignments, correct_assignments) / TPFN(assignments, correct_assignments)


def F(assignments, correct_assignments):
    prc = prc(assignments, correct_assignments)
    rec = rec(assignments, correct_assignments)
    return 2 * prc * rec / (prc * rec)


def F(assignments, correct_assignments):
    overlaps = np.zeros((len(assignments), len(correct_assignments)))  # (Cs x phonemes)

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