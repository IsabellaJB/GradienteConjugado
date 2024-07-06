import numpy as np

def himmelblau(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion


def gradiente(f, x, deltaX=1e-8):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)



def conjugate_gradient(funcion_objetivo, x, epsilon1, epsilon2, max_iterations):
    terminar = False
    xk = np.array(x, dtype=float)
    k = 0
    sk = -np.array(gradiente(funcion_objetivo, xk))  # initial search direction
    rk = np.array(gradiente(funcion_objetivo, xk))  # initial residual

    while not terminar:
        Ak = np.array(gradiente(funcion_objetivo, xk + sk))

        # Regularization to prevent division by zero
        denominator = np.dot(sk, Ak)
        if denominator == 0:
            denominator = 1e-8
        alpha = np.dot(rk, rk) / denominator

        xk = xk + alpha * sk

        rk = rk - alpha * Ak

        if np.linalg.norm(rk) < epsilon1 or k >= max_iterations:
            terminar = True
        else:
            beta = np.dot(rk, rk) / np.dot(gradiente(funcion_objetivo, xk - sk), gradiente(funcion_objetivo, xk - sk))
            sk = -rk + beta * sk

            k += 1

    return xk



x0 = np.array([1.0, 2.0])
epsilon1 = 1e-6
max_iterations = 100

# Execute the Conjugate Gradient method
x_opt = conjugate_gradient(himmelblau, x0, epsilon1, 0.0, max_iterations)

print("Optimal solution:", x_opt)