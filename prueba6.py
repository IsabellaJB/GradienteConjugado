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

# ---------------------------------- BUSQUEDA DE FIBONACCI ---------------------------------- 
def fibonacci_search(funcion, epsilon, a, b):
    fibs = [0, 1]
    while (b - a) / fibs[-1] > epsilon:
        fibs.append(fibs[-1] + fibs[-2])

    n = len(fibs) - 1
    k = n - 1

    x1 = a + fibs[k-1] / fibs[k] * (b - a)
    x2 = a + fibs[k] / fibs[k+1] * (b - a)
    f1 = funcion(x1)
    f2 = funcion(x2)
    
    while k > 1:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + fibs[k-1] / fibs[k] * (b - a)
            f2 = funcion(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + fibs[k-2] / fibs[k-1] * (b - a)
            f1 = funcion(x1)
        k -= 1

    if f1 < f2:
        return x1
    else:
        return x2


# ---------------------------------- DISTANCIA ORIGEN ---------------------------------- 
def distancia_origen(vector):
    return np.linalg.norm(vector)



# ---------------------------------- LINE SEARCH ---------------------------------- 
def line_search(f, xk, punto, metodo_busqueda, epsilon2=1e-3):
    def alpha_calcular(alpha):
        return f(xk - alpha * punto)
    
    alpha = metodo_busqueda(alpha_calcular, epsilon2, 0.0, 1.0)
    return alpha


# ---------------------------------- DESCENSO DE GRADIENTE ---------------------------------- 
def gradient_descent(f, grad_f, x0, epsilon=1e-3, max_iter=100, search_method='golden'):
    x = x0
    k = 0
    while k < max_iter:
        grad = grad_f(f, x)
        if np.linalg.norm(grad) < epsilon:
            break
        
        if search_method == 'golden':
            alpha = line_search(f, x, grad, busquedaDorada, epsilon)
        elif search_method == 'fibonacci':
            alpha = line_search(f, x, grad, fibonacci_search, epsilon)
        else:
            raise ValueError("Método de búsqueda no reconocido")
        
        x = x - alpha * grad
        k += 1
    return x, k



# Example usage
x0 = np.array([0, 0])
x, k = gradient_descent(funcion_objetivo, gradiente, x0, search_method='golden')
print(f"Minimum found at x = {x}, after {k} iterations.")
