import math
import numpy as np

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.clip(
        np.asarray(y_pred),
        min = eps,
        max = 1-eps
    )

    L = (-1) * (y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return L.tolist()