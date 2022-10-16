from enum import Enum

from scipy import linalg

inv = linalg.inv

from lib.conceptors import *
from lib.esn import ESN
from lib.helpers import *
from lib.plot import Plot

plot = Plot()
plot.new_window(3, 3)

########################################
# Hyperparameters

np.random.seed(0)
t_max = 1000
t_test = 50
t_washout = 500  # number of washout steps
aperture = 1
nb_trials = 1
adjacent_samples = 20

esn_params = {
    "in_dim": 1,
    "out_dim": 1,
    "N": 100,
    "W_in_scale": 1.5,
    "b_scale": .2,
    "spectral_radius": 1.5
}


########################################
# collect data

def gen_signal(n, period, amplitude):
    """
    Generates a sin wave
    """
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1 / period) * ts)
    return data


print("Generating signals")
p3 = gen_signal(t_max, 8, 1)
p1 = gen_signal(t_max, 14, 1)
p2 = gen_signal(t_max, 20, 1)
data = [p1, p2, p3]
# plot.add(p1[:t_test], "p1")
# plot.add_new(p2[:t_test], "p2")
# plot.add_new(p3[:t_test], "p3")
p_combined_incl_washout = np.concatenate((p1, p2[t_washout:], p3[t_washout:]))

#######################################
# init reservoir

esn = ESN(esn_params)

#######################################
# run the reservoir with the signal(s) p1, p2, and p3 and collect X

print("Running and loading reservoir with different signals")
X_truth = None
X_delay = None
P_truth = None
assignments_truth = [[] for _ in range(len(data))]
Cs_truth = []
t_tracker = 0
for i, signal in enumerate(data):
    X_local, X_delay_local = esn.run(signal, t_washout)
    if X_truth is None:
        X_truth = X_local
        X_delay = X_delay_local
        P_truth = signal[t_washout:]
        for j in range(len(data)):
            for _ in range(len(P_truth)):
                if j == i:
                    assignments_truth[j].append(1)
                else:
                    assignments_truth[j].append(0)
    else:
        # append new states X_local, X_delay (later used for loading), and pattern
        X_truth = np.concatenate((X_truth, X_local), axis=1)
        X_delay = np.concatenate((X_delay, X_delay_local), axis=1)
        P_truth = np.concatenate((P_truth, signal[t_washout:]))
        for j in range(len(data)):
            for _ in range(len(P_truth) - len(signal[t_washout:]), len(P_truth)):
                if j == i:
                    assignments_truth[j].append(1)
                else:
                    assignments_truth[j].append(0)
    Cs_truth.append(compute_c(X_local, aperture))

X_combined, X_combined_delay = esn.run(p_combined_incl_washout, t_washout)
p_combined = p_combined_incl_washout[t_washout:]
plot.add_new_conceptors_fit_plot(X_combined, Cs_truth, "Truth C")

#######################################
esn.load(X_combined, X_combined_delay, 1e-4)
esn.train_out_identity(X_combined, p_combined, 1e-2)


#######################################
# Generate mixed signal

class Method(Enum):
    """
    Transitions for generating interesting signals
    """
    RANDOM = 0
    RANGES = 1
    EQUAL_SPLIT = 2
    TRANSITIONS = 3


def assign_to_clusters(nb_points, nb_clusters, method=Method.EQUAL_SPLIT, limits=[]):
    """
    Assigns points to conceptors according to one of several assignment methods
    """
    assignments = [[] for _ in range(nb_clusters)]
    if method == Method.RANDOM or method == Method.EQUAL_SPLIT:
        points = [x for x in range(nb_points)]
        if method == Method.RANDOM:
            np.random.shuffle(points)
        for i in range(nb_clusters):
            assignments[i] = points[i * int(nb_points / nb_clusters):(i + 1) * int(nb_points / nb_clusters)]
    elif method == Method.RANGES:
        mark = 0
        for i in range(nb_points):
            if i in limits:
                mark += 1
            assignments[mark].append(i)
    return assignments


def assign_fuzzy_to_clusters(nb_points, nb_clusters, method=Method.EQUAL_SPLIT, limits=[], transition_time=0):
    """
    Distributes points over clusters: For each t, the assigned cluster gets 1
    """
    assignments = [[] for _ in range(nb_clusters)]
    if method == Method.TRANSITIONS:
        mean_length = int(nb_points / nb_clusters)
        current = -1
        for t in range(nb_points):
            if not t % mean_length:
                current += 1
            for i in range(nb_clusters):
                if i == current:
                    if t % mean_length > mean_length - transition_time and not i + 1 == nb_clusters:
                        assignments[i].append(
                            1 - (t % mean_length - (mean_length - transition_time)) * 1 / transition_time)
                    else:
                        assignments[i].append(1)
                elif i == current + 1 and t % mean_length > mean_length - transition_time:
                    assignments[i].append((t % mean_length - (mean_length - transition_time)) * 1 / transition_time)
                else:
                    assignments[i].append(0)
    else:
        mark = 0
        for t in range(nb_points):
            if method == Method.RANDOM:
                cluster_choice = np.random.randint(0, nb_clusters)
                for i, cluster in enumerate(assignments):
                    if i == cluster_choice:
                        cluster.append(1)
                    else:
                        cluster.append(0)
            elif method == Method.EQUAL_SPLIT:
                for i, cluster in enumerate(assignments):
                    if i == int(t * nb_clusters / nb_points):
                        cluster.append(1)
                    else:
                        cluster.append(0)
            elif method == Method.RANGES:
                if t in limits:
                    mark += 1
                for i, cluster in enumerate(assignments):
                    if i == mark:
                        cluster.append(1)
                    else:
                        cluster.append(0)
    return assignments


def compute_memberships(X, Cs):
    """
    For every point, compute how much it corresponds each of the conceptors 
    """
    nb_points = X.shape[1]
    assignments = [[] for _ in Cs]
    for t in range(nb_points):
        # e_by_C = normalized_evidences_by_conceptor(X[:,t], Cs)
        e_by_C = best_mus(X[:, t], Cs)
        for i in range(len(assignments)):
            assignments[i].append(e_by_C[i])
    return assignments


assignments_fuzzy = assign_fuzzy_to_clusters(t_max, 3, transition_time=100)
# Generate a morphed signal - unused at the moment
p_generated, X_regen = esn.generate(Cs_truth, assignments_fuzzy, t_max, True)


#######################################
# C-means
def cmeans(X, nb_conceptors, method, limits, aperture, max_epochs=100, plot_progress=False):
    """
    Cmeans algorithm, adapted to conceptors
    """
    print("C-means")
    # Initial assignments and initial conceptors
    nb_points = X.shape[1]
    new_assignments = assign_fuzzy_to_clusters(nb_points, nb_conceptors, method, limits)
    # Training loop
    for epoch in range(max_epochs):
        print("epoch:", epoch)
        # recompute centroids based on subset of assigned state

        Cs = [compute_c(X[:, assignments], aperture) for assignments in get_assignments(cuts, c_map)]
        Ns = Ns_from_Cs(Cs)
        # print("- optimizing +")
        Cs = optimize_apertures(Cs)
        # print("- optimizing -")
        Ns = optimize_apertures(Ns)
        plot.add_new_assignment_plot(get_assignments(cuts, c_map), label="C")

        Cs = [compute_c(X, aperture, assignments) for assignments in new_assignments]
        # Cs = optimize_apertures(Cs)
        Cs = normalize_apertures(Cs)

        if plot_progress:
            plot.add_new_assignment_plot(new_assignments, "", True)

            # plot.add_new_conceptors_fit_plot(X, Cs, "K-means epoch:"+str(epoch)+", C")
        # recompute assignments by find the closest conceptor for each of the state points
        old_assignments = new_assignments.copy()

        new_assignments = compute_memberships(X, Cs)

        # stop if converged
        stop = True
        eps = 0.001  # Convergence threshold
        for i in range(nb_conceptors):
            for t in range(nb_points):
                if abs(new_assignments[i][t] - old_assignments[i][t]) > eps:
                    stop = False
                    break
            if stop:
                break
        if stop:
            print("Converged")
            return Cs, new_assignments

    return Cs, new_assignments


#######################################

def experiment(method, limits=[]):
    print("METHOD:", method)
    mce_delta_mean = 0
    mce_mean = 0
    sim_mean = 0
    nrmse_mean = 0
    for trial in range(nb_trials):
        # cluster into as many conceptors as patterns
        Cs_cmeans, assignments_kmeans = cmeans(X_combined, nb_conceptors=3, method=method, limits=limits,
                                               aperture=aperture, max_epochs=100,
                                               plot_progress=(trial == nb_trials - 1))

        print(similarity_c(Cs_cmeans[0], Cs_cmeans[1]))
        print(similarity_c(Cs_cmeans[2], Cs_cmeans[1]))
        print(similarity_c(Cs_cmeans[0], Cs_cmeans[2]))
        ### Testing

        # Mean combined evidence
        mce_kmeans = weighted_mean_combined_evidence(X_combined, Cs_cmeans, assignments_kmeans)
        mce_truth = weighted_mean_combined_evidence(X_truth, Cs_truth, assignments_truth)
        mce_mean += mce_kmeans
        mce_delta_mean += (mce_truth - mce_kmeans) / nb_trials

        # Distance
        sim = max_similarity(Cs_cmeans, Cs_truth)
        sim_mean += sim / nb_trials

        # Regeneration
        p_regen, _ = esn.generate(Cs_cmeans, assignments_kmeans, len(p_combined), fuzzy=True)
        pp, _ = esn.generate(Cs_truth, assignments_truth, len(p_combined), fuzzy=True)
        nrmse = walking_NRMSE(p_combined, p_regen, 50, 15)
        nrmse_mean += nrmse / nb_trials

        if trial == nb_trials - 1:
            plot.add_new_assignment_plot(assignments_kmeans, "Kmeans assignments to C ")
            plot.finalize(
                title='Aperture=' + str(aperture) + ', N=' + str(esn.N) + ', Spec Rad=' + str(esn.spectral_radius))

            plot.new_window(2, 2)
            plot.add_new(p_combined, "Input signal without initial washout period")
            plot.add_new(p_regen, "Regenerated signal")
            plot.add_new(pp, "Regenerated truth signal")
            plot.finalize()

    print("Mean MCE:", mce_mean)
    print("Mean Delta MCE:", mce_delta_mean)
    print("Mean Maximum Similarity:", sim_mean)
    print("Mean Regeneration NRMSE:", nrmse_mean)


# experiment(Method.RANDOM)
experiment(Method.EQUAL_SPLIT)
# experiment(Method.RANGES, [300, 800])
