# import numpy as np

# def gradient_conjugate(f, grad_f, x0, epsilon1, epsilon2, epsilon3):
#   """
#   Implements the Gradient Conjugate algorithm for minimizing a function.

#   Args:
#     f: The function to minimize.
#     grad_f: The gradient of the function.
#     x0: The initial point.
#     epsilon1: Termination parameter for line search.
#     epsilon2: Termination parameter for step size.
#     epsilon3: Termination parameter for gradient norm.

#   Returns:
#     The minimizer x, the function value at the minimizer f(x), and the number of iterations.
#   """

#   x = x0
#   k = 0
#   s = -grad_f(x)
  
#   while True:
#     # Step 3: Line search
#     alpha = line_search(f, x, s, epsilon1)
#     x = x + alpha * s

#     # Step 4: Calculate gradient and update search direction
#     grad = grad_f(x)
#     if k > 0:
#       beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
#       s = -grad + beta * s
#     else:
#       s = -grad
#     grad_prev = grad

#     # Step 6: Check termination conditions
#     if np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev) <= epsilon2 or np.linalg.norm(grad) <= epsilon3:
#       break

#     k += 1
#     x_prev = x

#   return x, f(x), k

# def line_search(f, x, s, epsilon):
#   """
#   Performs a line search to find the optimal step size.

#   Args:
#     f: The function to minimize.
#     x: The current point.
#     s: The search direction.
#     epsilon: Termination parameter.

#   Returns:
#     The optimal step size alpha.
#   """

#   alpha = 1.0
#   while True:
#     if f(x + alpha * s) < f(x) - epsilon * alpha * np.dot(grad_f(x), s):
#       break
#     alpha /= 2

#   return alpha

# # Example usage
# def f(x):
#   return x[0]**2 + x[1]**2

# def grad_f(x):
#   return np.array([2*x[0], 2*x[1]])

# x0 = np.array([1, 1])
# epsilon1 = 1e-6
# epsilon2 = 1e-6
# epsilon3 = 1e-6

# x_min, f_min, k = gradient_conjugate(f, grad_f, x0, epsilon1, epsilon2, epsilon3)

# print(f"Minimizer: {x_min}")
# print(f"Function value at minimizer: {f_min}")
# print(f"Number of iterations: {k}")

import numpy as np

def gradient_conjugate(f, grad_f, x0, epsilon1, epsilon2, epsilon3):
    """
    Implements the Gradient Conjugate algorithm for minimizing a function.

    Args:
      f: The function to minimize.
      grad_f: The gradient of the function.
      x0: The initial point.
      epsilon1: Termination parameter for line search.
      epsilon2: Termination parameter for step size.
      epsilon3: Termination parameter for gradient norm.

    Returns:
      The minimizer x, the function value at the minimizer f(x), and the number of iterations.
    """
    
    x = x0
    x_prev = x0  # Initialize x_prev
    k = 0
    s = -grad_f(x)
    
    while True:
        # Step 3: Line search
        alpha = line_search(f, grad_f, x, s, epsilon1)
        x = x + alpha * s

        # Step 4: Calculate gradient and update search direction
        grad = grad_f(x)
        if k > 0:
            beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
            s = -grad + beta * s
        else:
            s = -grad
        grad_prev = grad

        # Step 6: Check termination conditions
        if np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev) <= epsilon2 or np.linalg.norm(grad) <= epsilon3:
            break

        k += 1
        x_prev = x

    return x, f(x), k

def line_search(f, grad_f, x, s, epsilon):
    """
    Performs a line search to find the optimal step size.

    Args:
      f: The function to minimize.
      grad_f: The gradient of the function.
      x: The current point.
      s: The search direction.
      epsilon: Termination parameter.

    Returns:
      The optimal step size alpha.
    """

    alpha = 1.0
    while True:
        if f(x + alpha * s) < f(x) - epsilon * alpha * np.dot(grad_f(x), s):
            break
        alpha /= 2

    return alpha

# Example usage
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1, 1])
epsilon1 = 1e-6
epsilon2 = 1e-6
epsilon3 = 1e-6

x_min, f_min, k = gradient_conjugate(f, grad_f, x0, epsilon1, epsilon2, epsilon3)

print(f"Minimizer: {x_min}")
print(f"Function value at minimizer: {f_min}")
print(f"Number of iterations: {k}")
