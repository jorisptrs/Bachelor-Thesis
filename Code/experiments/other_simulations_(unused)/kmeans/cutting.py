from scipy import linalg

inv = linalg.inv
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from lib.conceptors import *
from lib.esn import ESN
from lib.plot import Plot
from lib.conceptors import *

plot = Plot()
plot.new_window(1, 2)

########################################
# Hyperparameters

np.random.seed(0)
t_max = 300
t_washout = 0
aperture = 5

esn_params = {
    "in_dim": 1,
    "out_dim": 1,
    "N": 50,
    "W_in_scale": 1.5,
    "b_scale": .2,
    "spectral_radius": 1.5
}


########################################
# collect dataset

def gen_signal(n, period, amplitude):
    """
    Generates a sin wave
    """
    ts = np.arange(n)
    data = amplitude * np.sin(2 * np.pi * (1 / period) * ts)
    return data


print("Generating signals")
p1 = gen_signal(t_max, 8, 1)
p2 = gen_signal(t_max, 14, 1)
p3 = gen_signal(t_max, 20, 1)
data = [p1, p2, p3]
p = np.concatenate((p1, p3, p2, p3))
plot.add_new(p)

#######################################
# init reservoir

esn = ESN(esn_params)

X, _ = esn.run_X(p, p.size, t_washout)


def cutt(nb_points, nb_clusters):
    """
    Assigns points to conceptors according to one of several assignment methods
    """
    return [
               [int(i * nb_points / nb_clusters), int((i + 1) * nb_points / nb_clusters) - 1] for i in
               range(nb_clusters)
           ], list(range(nb_clusters))


def get_cut(x, cuts):
    for i, cut in enumerate(cuts):
        if x >= cut[0] and x <= cut[1]:
            return i


def c_of_cut(C_map, cut):
    return C_map[cut]


def in_cut(x, i, cuts):
    return get_cut(x, cuts) == i


def get_assignments(cuts, c_map):
    nb_clusters = max(c_map) + 1
    res = [[] for _ in range(nb_clusters)]
    for c in range(nb_clusters):
        for i, cut in enumerate(cuts):
            if c == c_map[i]:
                res[c] += list(range(cut[0], cut[1] + 1))
    return res


#######################################
# K-means
def cutting(X, nb_conceptors):
    nb_points = X.shape[1]
    cuts, c_map = cutt(nb_points, nb_conceptors)

    for t in range(3 * nb_points):
        if t % 50 == 0:
            Cs = [compute_c(X[:, assignments], aperture) for assignments in get_assignments(cuts, c_map)]
            Ns = Ns_from_Cs(Cs)
            # print("- optimizing +")
            Cs = optimize_apertures(Cs)
            # print("- optimizing -")
            Ns = optimize_apertures(Ns)
            plot.add_new_assignment_plot(get_assignments(cuts, c_map), label="C")
            print(t)
        x = np.random.randint(0, nb_points)
        alpha = (1 - t / (3 * nb_points))
        cut_old = get_cut(x, cuts)
        c_old = c_of_cut(c_map, cut_old)
        Es = evidences_for_Cs_z(X[:, x], Cs, Ns)
        Es = np.array(Es) / sum(Es)

        Es_for_cuts = []
        for cut in range(len(cuts)):
            if cut == cut_old or in_cut(x - 1, cut, cuts) or in_cut(x + 1, cut, cuts):
                Es_for_cuts.append(Es[c_of_cut(c_map, cut)])
            elif cut != cut_old and (in_cut(x - 1, cut_old, cuts) or x == 0) and (
                    in_cut(x + 1, cut_old, cuts) or x == nb_points - 1):
                Es_for_cuts.append(Es[c_of_cut(c_map, cut)] * (alpha / 2))
            else:
                Es_for_cuts.append(Es[c_of_cut(c_map, cut)] * alpha)

        c_new = c_of_cut(c_map, np.argmax(np.array(Es_for_cuts)))
        # add to c_new
        if c_old == c_new:
            continue
        elif (in_cut(x - 1, cut_old, cuts) or x == 0) and (in_cut(x + 1, cut_old, cuts) or x == nb_points - 1):
            # create island
            temp = cuts[cut_old][1]
            cuts[cut_old][1] = x - 1
            cuts.insert(cut_old + 1, [x, x])
            cuts.insert(cut_old + 2, [x + 1, temp])
            c_map.insert(cut_old + 1, c_new)
            c_map.insert(cut_old + 2, c_old)
        elif x == cuts[cut_old][1]:
            if cut_old < len(cuts) - 1 and c_new == c_of_cut(c_map, cut_old + 1):
                # move forth
                cuts[cut_old][1] -= 1
                cuts[cut_old + 1][0] -= 1
            else:
                cuts[cut_old][1] -= 1
                cuts.insert(cut_old + 1, [x, x])
                c_map.insert(cut_old + 1, c_new)
        elif x == cuts[cut_old][0]:
            if cut_old > 0 and c_new == c_of_cut(c_map, cut_old - 1):
                # move back
                cuts[cut_old][0] += 1
                cuts[cut_old - 1][1] += 1
            else:
                cuts[cut_old][0] += 1
                cuts.insert(cut_old, [x, x])
                c_map.insert(cut_old, c_new)
        else:
            print("huh?")

        # a = get_assignments(cuts, c_map)
        lb = nb_points / nb_conceptors  # len(a[c_old])
        Cs[c_old] = remove_instance(Cs[c_old], X[:, x], lb, 100000)
        # lb = 1 / len(a[c_new])
        Cs[c_new] = add_instance(Cs[c_new], X[:, x], lb, 100000)

        cuts = [cut for cut in cuts if cut != []]
    return cuts, c_map


cuts, c_map = cutting(X, 3)
print(cuts)
print(get_assignments(cuts, c_map))
plot.add_new_assignment_plot(get_assignments(cuts, c_map), label="C")
plot.finalize()
