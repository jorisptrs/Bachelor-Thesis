import numpy as np


def smoothed(vals, d=20):
    return np.convolve(np.array(vals), np.ones(d), 'valid') / d


# by changing the coorinates of the above you can repeat this for the y axis too
def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')
