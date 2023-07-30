import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from classifier import Classifier
from debug import debug_print
from experiments.exp2_below_phoneme_clustering.kmeans.kmeans import KMeans
from experiments.exp2_below_phoneme_clustering.kmeans.point import Point


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
    return [NMI(a, correct_assignments) for a in assignment_list]


def get_overlaps(assignments, correct_assignments):
    overlaps = np.zeros((len(correct_assignments), 1))
    for j, assignment in enumerate(assignments):
        for i, correct_assignment in enumerate(correct_assignments):
            for p in assignment:
                overlaps[j] += p in correct_assignment
    return overlaps

def get_heat_map(ps, sim_func, zero_diag=True):
    heat_map = np.zeros((len(ps), len(ps)))

    for x in range(len(ps)):
        for y in range(0, x + (not zero_diag)):
            sim = sim_func(ps[x], ps[y])
            heat_map[x, y] = sim
            heat_map[y, x] = sim

    return heat_map

def match_clusters_to_classes(clusters, ground_truth, phonemes):
    scores = []
    for i, cluster in enumerate(clusters):
        for j, true_class in enumerate(ground_truth):
            match_score = len([p for p in cluster if p in true_class])
            scores.append((match_score, i, j))
    scores.sort(reverse=True)

    # Keep track of assignments
    cluster_labels = [None] * len(clusters)
    class_assigned = [False] * len(ground_truth)

    for score, i, j in scores:
        if cluster_labels[i] is not None or class_assigned[j]:
            continue
        cluster_labels[i] = phonemes[j]
        class_assigned[j] = True

    if None in cluster_labels:
        debug_print("Not all clusters could be assigned to a class")

    return cluster_labels

def helper(clusters, labels):
    features_train, labels_train = [], []
    for i, cluster in enumerate(clusters):
        features_train += Point.get_signals(cluster)
        labels_train += [labels[i]] * len(cluster)
    return features_train, labels_train

def train_and_test_clas(features_train, labels_train, features_test, labels_test):
    n_mels = 14
    clas = Classifier(W_in_scale=1.1,
                      b_scale=.6,
                      spectral_radius=2.57,
                      weights=.1)

    _ = clas.fit(features_train, labels_train, **{
        "n_mels": n_mels,
        "XorZ": "X",
        "N": 100,
        "cache": False
    })
    return clas.score(features_test, labels_test)

def clas_acc(clusters, correct_clusters, phonemes, features_test, labels_test):
    matched_labels = match_clusters_to_classes(clusters, correct_clusters, phonemes)
    features_train, labels_train = helper(clusters, matched_labels)
    return train_and_test_clas(features_train, labels_train, features_test, labels_test)

def clas_acc_from_list(cluster_hist, correct_clusters, phonemes, features_test, labels_test):
    return [clas_acc(clusters, correct_clusters, phonemes, features_test, labels_test) for clusters in cluster_hist]
