
# implementación del método simplex 

import numpy as np
print("METODO SIMPLEX")

def leer_datos():
    # Función para ingresar los datos del problema
    print("Ingrese los coeficientes de la función objetivo (separados por espacios):")
    c = list(map(float, input().split()))
    print("Ingrese el número de restricciones:")
    m = int(input())
    A = []
    b = []
    for i in range(m):
        print(f"Ingrese los coeficientes de la restricción {i+1} (separados por espacios):")
        fila = list(map(float, input().split()))
        A.append(fila)
        print(f"Ingrese el valor del lado derecho de la restricción {i+1}:")
        b.append(float(input()))
    return np.array(c), np.array(A), np.array(b)

def simplex(c, A, b):
    m, n = A.shape
    # Construir tabla simplex inicial
    tabla = np.zeros((m+1, n+m+1))
    # Agregar A y variables de holgura
    tabla[:m, :n] = A
    tabla[:m, n:n+m] = np.eye(m)
    tabla[:m, -1] = b
    # Función objetivo
    tabla[-1, :n] = -c

    while True:
        print("Tabla Simplex Actual:")
        print(tabla)
        # Verificar si hay valores negativos en la fila de función objetivo (maximización)
        if all(tabla[-1, :-1] >= 0):
            print("Solución óptima alcanzada.")
            break
        # Columna pivote (entrada)
        j = np.argmin(tabla[-1, :-1])
        # Cocientes para determinar fila pivote (salida)
        if all(tabla[:-1, j] <= 0):
            print("Solución no acotada")
            return None
        cocientes = np.where(tabla[:-1, j] > 0, tabla[:-1, -1] / tabla[:-1, j], np.inf)
        i = np.argmin(cocientes)
        # Normalizar fila pivote
        tabla[i, :] /= tabla[i, j]
        # Operar otras filas
        for k in range(m+1):
            if k != i:
                tabla[k, :] -= tabla[k, j] * tabla[i, :]
        # Pedir datos nuevamente si la función objetivo tiene valores negativos
        if any(tabla[-1, :-1] < 0):
            print("La fila de la función objetivo aún tiene valores negativos. Puede modificar los datos si desea repetir.")
            opcion = input("¿Desea ingresar nuevos datos? (s/n): ").lower()
            if opcion == 's':
                c, A, b = leer_datos()
                return simplex(c, A, b)
    print("Solución óptima encontrada:")
    print("Valores de variables (incluyendo holgura):", tabla[:-1, -1])
    print("Valor máximo de la función objetivo:", tabla[-1, -1])

# Ejecución
c, A, b = leer_datos()
simplex(c, A, b)