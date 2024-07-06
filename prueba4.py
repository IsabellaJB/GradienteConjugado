import numpy as np

def himmelblau(x):
  return (x[0] + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def gradient_descent(f, x0, epsilon=1e-3, max_iter=100):
  x = x0
  d = -np.gradient(f(x))
  k = 0
  while k < max_iter:
    alpha = line_search(f, x, d)
    x = x + alpha * d
    if np.linalg.norm(np.gradient(f(x))) < epsilon:
      break
    d = -np.gradient(f(x)) + (np.linalg.norm(np.gradient(f(x)))**2 / np.linalg.norm(np.gradient(f(x - alpha * d)))**2) * d
    k += 1
  return x, k

def line_search(f, x, d):
  """Line search using golden section search."""
  a = 0
  b = 1
  phi = (1 + np.sqrt(5)) / 2
  while abs(b - a) > 1e-3:
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    if f(x + c * d) < f(x + d * d):
      b = d
    else:
      a = c
  return (a + b) / 2

# Example usage
x0 = np.array([0, 0])
x, k = gradient_descent(himmelblau, x0)
print(f"Minimum found at x = {x}, after {k} iterations.")