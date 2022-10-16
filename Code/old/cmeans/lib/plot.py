from math import floor

import matplotlib.pyplot as plt
from lib.conceptors import *
from lib.helpers import *

"""
Problem-specific plotting
"""


class Plot:
    def __init__(self):
        self.figCnt = 0

    def new_window(self, ymax, xmax):
        # plt.figure(self.figCnt)
        self.ymax = ymax
        self.xmax = xmax
        self.cnt = -1
        self.fig, self.ax = plt.subplots(self.ymax, self.xmax)
        self.figCnt += 1

    def add(self, y, label="No Label"):
        y_idx = self.cnt % self.ymax
        x_idx = int(floor(self.cnt / self.ymax))
        self.ax[y_idx, x_idx].plot(y, label=label)
        self.ax[y_idx, x_idx].legend()

    def inc(self):
        if self.cnt < self.xmax * self.ymax:
            self.cnt += 1

    def add_new(self, y, label="No Label"):
        self.inc()
        self.add(y, label)

    def add_new_assignment_plot(self, assignments, label="No Label", fuzzy=True):
        self.inc()
        for idx, ts in enumerate(assignments):
            if not fuzzy:
                y = []
                for t in range(max([max(ts) for ts in assignments])):
                    if t in ts:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                y = ts
            self.add(smoothed(y, 1), label + str(idx))

    def add_new_conceptors_fit_plot(self, X, Cs, label):
        """
        Plots, for each time step t, how well each conceptor in Cs matches the state x(t)
        """
        self.inc()
        collection, _ = test(X, Cs, "PROP")
        for i, vals in enumerate(collection):
            # walking average of d
            self.add(smoothed(vals, 1), label=label + str(i))

    def finalize(self, title=""):
        self.fig.suptitle(title, fontsize=16)
        plt.show(block=True)
