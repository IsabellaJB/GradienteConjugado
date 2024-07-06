import numpy as np

def himmelblau(x):
    """Himmelblau function."""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    """Gradient of the Himmelblau function."""
    dx = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    dy = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return np.array([dx, dy])

def gradient_descent(f, grad_f, x0, epsilon=1e-3, max_iter=100):
    """Gradient Descent."""
    x = x0
    k = 0
    while k < max_iter:
        grad = grad_f(x)
        if np.linalg.norm(grad) < epsilon:
            break
        alpha = line_search(f, grad, x, -grad)
        x = x - alpha * grad
        k += 1
    return x, k

def line_search(f, grad, x, d, alpha_init=1.0, tau=0.5, c=1e-4):
    
    alpha = alpha_init
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(grad, d):
        alpha *= tau
    return alpha

# Example usage
x0 = np.array([0, 0])
x, k = gradient_descent(himmelblau, himmelblau_grad, x0)
print(f"Minimum found at x = {x}, after {k} iterations.")
