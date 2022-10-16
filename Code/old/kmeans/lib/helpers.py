import numpy as np


def ridge_regression(X, Y, reg_param):
    """
    Ridge regression solving W x = y
    """
    XTX = X.T @ X
    return (np.linalg.inv(XTX + reg_param * np.eye(XTX.shape[0])) @ X.T @ Y).T


def smoothed(vals, d=20):
    return np.convolve(np.array(vals), np.ones(d), 'valid') / d
