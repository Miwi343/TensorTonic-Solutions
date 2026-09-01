def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # find gradient
    def gradient(x):
        """
        finds gradient at x
        know gradient of ax^2 + bx + c -> 2ax + b
        """
        return 2*a*x + b

    # step in negative direction of gradient
    # find gradient and then step
    for _ in range(steps):
        grad = gradient(x0)
        x0 = x0 - grad * lr

    return x0