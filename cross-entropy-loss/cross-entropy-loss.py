import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    N = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[np.arange(N), y_true]

    L = (-1/N) * np.sum(np.log(y_pred))
    return L

    

    