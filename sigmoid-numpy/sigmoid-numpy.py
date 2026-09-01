import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """

    # need to pass an activition to each element
    vector = np.array(x, dtype=float)

    return 1 / (1 + np.exp(-vector))
