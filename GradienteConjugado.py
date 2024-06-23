import numpy as np

def himmelblau(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

def gradiente(f, x, deltaX=0.001):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)

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

    return x

def line_search(f, x, s, e1):
    alpha = 1.0
    while f(x + alpha * s) > f(x):
        alpha *= 0.5
        if alpha < e1:
            break
    return alpha

def redondear(arreglo):
    lita = []
    for valor in arreglo:
        v = round(valor, 1)
        lita.append(v)
    return(lita)

x0 = np.array([2.0, 3.0])
e1 = 0.001
e2 = 0.001
e3 = 0.001

result = gradiente_conjugado(himmelblau, x0, e1, e2, e3)
print("Resultado optimizado:", result)


nuevos = redondear(result)
print(nuevos)





