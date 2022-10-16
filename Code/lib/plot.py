from math import floor

import matplotlib.pyplot as plt
from lib.conceptors import *
from lib.helpers import *

"""
Problem-specific plotting
"""


class Plot:
    def __init__(self):
        plt.rcParams["figure.autolayout"] = True
        self.figCnt = 0

    def new_window(self, ymax, xmax, figsize=None):
        # if figsize:
        #    plt.figure(self.figCnt, figsize=figsize)
        # else:
        #    plt.figure(self.figCnt)
        self.ymax = ymax
        self.xmax = xmax
        self.cnt = -1
        self.fig, self.ax = plt.subplots(self.ymax, self.xmax)
        self.figCnt += 1
        self.d = 1 if self.ymax == 1 or self.xmax == 1 else 2
        self.legend_tracker = np.zeros((ymax, xmax)) if self.d == 2 else np.zeros(max(ymax, xmax))

    def add(self, y, label="No Label"):
        if self.ymax == 1 or self.xmax == 1:
            self.ax[self.cnt].plot(y, label=label)
            self.legend_tracker[self.cnt] += 1
        else:
            y_idx = self.cnt % self.ymax
            x_idx = int(floor(self.cnt / self.ymax))
            self.ax[y_idx, x_idx].plot(y, label=label)
            self.legend_tracker[y_idx, x_idx] += 1

    def inc(self):
        if self.cnt < self.xmax * self.ymax - 1:
            self.cnt += 1

    def add_new(self, y, label="No Label"):
        self.inc()
        self.add(y, label)

    def add_new_assignment_plot(self, assignments, label="No Label", fuzzy=False, length=0):
        self.inc()
        self.ax[self.cnt].clear()
        for idx, ts in enumerate(assignments):
            if not fuzzy:
                y = []
                if length == 0:
                    length = max([max(ts) for ts in assignments if ts != []])
                for t in range(length):
                    if t in ts:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                y = ts
            self.add(smoothed(y, 1), label + str(idx))
        plt.pause(.2)

    def add_new_conceptors_fit_plot(self, X, Cs, Ns=None, label="", labels=""):
        """
        Plots, for each time step t, how well each conceptor in Cs matches the state x(t)
        """
        # self.inc()
        if Ns:
            collection = evidences_for_Cs(X, Cs, Ns)
        else:
            collection, _ = test(X, Cs, "PROP")
        if labels:
            for vals, label in zip(collection, labels):
                # walking average of d
                self.add(smoothed(vals, 3), label=label)
        else:
            for i, vals in enumerate(collection):
                # walking average of d
                self.add(smoothed(vals, 3), label=label + str(i))

    def finalize(self, title=""):
        self.fig.suptitle(title, fontsize=16)
        # for y in range(self.ymax):
        #     for x in range(self.xmax):
        #         if self.d == 2 and self.legend_tracker[y, x]:
        #             self.ax[y,x].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #             ncol=self.legend_tracker[y, x], fancybox=True, shadow=True)
        #         elif self.legend_tracker[max(y,x)]:
        #             self.ax[max(y,x)].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #             ncol=self.legend_tracker[max(y,x)], fancybox=True, shadow=True)
        plt.show()
