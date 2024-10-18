from matplotlib import gridspec
import matplotlib.pyplot as plt

from env_variables import default_fig_size
from lib.conceptors import *
from lib.plotting.plotting_helpers import smoothed

"""
Problem-specific plotting
Used by clustering_below_phon
"""


class Plot:
    def __init__(self, x=10, y=10):
        plt.rcParams["figure.autolayout"] = True
        self.fig = plt.figure(figsize=(y, x))
        self.cnt = 0
        self.new_ax = None

    def add(self, y, x=None, label=None):
        if label is not None:
            if x is not None:
                self.new_ax.plot(x, y, label=label)
            else:
                self.new_ax.plot(y, label=label)
            #self.new_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            #                   ncol=2, fancybox=True, shadow=True,
            #                   handleheight=2.4, labelspacing=0.05)
            self.new_ax.legend()
        else:
            if x is None:
                self.new_ax.plot(y)
            else:
                self.new_ax.plot(x,y)

    def inc(self, ylabel="", xlabel="", title=""):
        """Plots the data to a new subplot at the bottom."""
        self.cnt += 1
        gs = gridspec.GridSpec(self.cnt, 1)

        # Reposition existing subplots
        for i, ax in enumerate(self.fig.axes):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])

        # Add new subplot
        self.new_ax = self.fig.add_subplot(gs[self.cnt-1])
        self.new_ax.set_xlabel(xlabel, fontsize=16)
        self.new_ax.set_ylabel(ylabel, fontsize=16)
        self.new_ax.set_title(title, fontsize=18)

    @staticmethod
    def assignment_to_function_repr(assignment, length):
        y = []
        for idx in range(length):
            if idx in assignment:
                y.append(1)
            else:
                y.append(0)
        return y

    def add_new(self, y, label="No Label"):
        self.inc()
        self.add(y, label=label)

    def plot_progress(self, NMI_list, silh_list):
        plt.figure()
        plt.plot(NMI_list, color="orange", label="GCHC NMI")
        plt.axhline(y=nmi_baseline, color='orange', linestyle='dashdot', label="Baseline NMI")
        plt.axhline(y=nmi_truth, color='orange', linestyle='dashed', label="Dataset NMI")
        plt.plot(silh_list, color="blue", label="GCHC SC")
        plt.axhline(y=sil_baseline, color='blue', linestyle='dashdot', label="Baseline SC")
        plt.axhline(y=sil_truth, color='blue', linestyle='dashed', label="Dataset SC")
        plt.legend()

    def add_new_assignment_plot(self, assignments, labels=[], smoothness=5, xlabel="", ylabel=""):
        self.inc(ylabel, xlabel)
        length = max([max(cluster) for cluster in assignments if cluster])
        for idx, assignment in enumerate(assignments):
            y = Plot.assignment_to_function_repr(assignment, length)
            if max(assignment) > 0:
                if labels is None:
                    self.add(y=smoothed(y, smoothness))
                elif not labels:
                    self.add(y=smoothed(y, smoothness), label=idx)
                else:
                    self.add(y=smoothed(y, smoothness), label=str(labels[idx]))

    def add_new_conceptors_fit_plot(self, X, Cs, Ns=None, label="", labels="", smoothness=3):
        """
        Plots, for each time step t, how well each conceptor in Cs matches the state x(t)
        """
        self.inc()
        if Ns:
            collection = evidences_for_Cs(X, Cs, Ns)
        else:
            collection, _ = test(X, Cs, "PROP")
        if labels:
            for vals, label in zip(collection, labels):
                # walking average of d
                self.add(smoothed(vals, smoothness), label=label)
        else:
            for i, vals in enumerate(collection):
                # walking average of d
                self.add(smoothed(vals, smoothness), label=label + str(i))

    def finalize(self, title=""):
        self.fig.suptitle(title, fontsize=16)
        plt.show()
