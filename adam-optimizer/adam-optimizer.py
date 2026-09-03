import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    np_param = np.array(param, dtype=float)
    np_grad = np.array(grad, dtype=float)

    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    
    # First moment
    m_new = beta1 * m + (1-beta1) * np_grad

    # second moment
    v_new = beta2 * v + (1-beta2) * (np_grad ** 2)

    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    param_new = np_param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return (param_new, m_new, v_new)