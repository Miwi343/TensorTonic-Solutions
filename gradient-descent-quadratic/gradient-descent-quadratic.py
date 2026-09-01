def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # find gradient
    def gradient(x):
        """
        finds gradient at x
        """
        return 2*a*x + b

    # step in negative direction of gradient
    for _ in range(steps):
        grad = gradient(x0)
        x0 = x0 - grad * lr

    return x0