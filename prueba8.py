import numpy as np
import math

# ---------------------------------- FUNCION OBJETIVO ---------------------------------- 
def funcion_objetivo(arreglo):
    x = arreglo[0]
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

# ---------------------------------- GRADIENTE ---------------------------------- 
def gradiente(funcion, x, delta=0.001):
    derivadas = []
    for i in range(len(x)):
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        copia[i] = x[i] - delta
        valor2 = funcion(copia)
        derivada = (valor1 - valor2) / (2 * delta)
        derivadas.append(derivada)
    return np.array(derivadas)

# ---------------------------------- BUSQUEDA DORADA ---------------------------------- 
def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w, a, b):
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon, a, b):
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

# ------------------------------------ Gradiente Conjugado ------------------------------------ 
def gradiente_conjugado(f, x0, e1, e2, e3):
    x = x0
    grad = gradiente(f, x)
    s = -grad
    k = 0

    while True:
        alpha = line_search(f, x, s, e1)
        x_next = x + alpha * s
        grad_next = gradiente(f, x_next)

        if np.linalg.norm(x_next - x) / (np.linalg.norm(x) + 1e-8) <= e2 or np.linalg.norm(grad_next) <= e3:
            break

        beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)
        s = -grad_next + beta * s

        x = x_next
        grad = grad_next
        k += 1

    return x, f(x), k

def line_search(f, x, s, e1):
    def alpha_funcion(alpha):
        return f(x + alpha * s)
    
    return busquedaDorada(alpha_funcion, e1, 0.0, 1.0)

# Example usage
x0 = np.array([1, 1])
epsilon1 = 1e-6
epsilon2 = 1e-6
epsilon3 = 1e-6

x_min, f_min, k = gradiente_conjugado(funcion_objetivo, x0, epsilon1, epsilon2, epsilon3)

print(f"Minimizer: {x_min}")
print(f"Function value at minimizer: {f_min}")
print(f"Number of iterations: {k}")
