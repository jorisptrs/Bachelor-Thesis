
def shift(signal, phase):
    for _ in range(phase):
        signal = signal[-1] + signal[:-1]
    return signal

def ridge_regression(X, Y, reg_param):
    """
    Ridge regression solving W x = y
    """
    XTX = X.T @ X
    return (np.linalg.inv(XTX + reg_param * np.eye(XTX.shape[0])) @ X.T @ Y).T
