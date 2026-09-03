import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n, d = X.shape
    w = np.zeros(d, dtype = float)
    b = 0.0

    print(f"Y shape: {y.shape}")
    y = y[:,]
    print(f"Y shape: {y.shape}")

    print("Starting training")

    for _ in range(steps):
        preds = _sigmoid( X @ w + b )
        grad_w = (1/n) * (X.T @ (preds - y))
        grad_b = (1/n) * np.sum(preds - y)
        
        w = w - lr * grad_w
        b = b - lr * grad_b

    return (w, b)