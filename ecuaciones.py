import numpy as np

def gaussian_elimination(A, B):
    augmented_matrix = np.hstack([A, B])
    n_rows, n_cols = augmented_matrix.shape
    for i in range(n_rows):
        pivot_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
        pivot_element = augmented_matrix[i, i]
        if pivot_element == 0:
            raise ValueError("La matriz no es invertible")
        augmented_matrix[i] = augmented_matrix[i] / pivot_element
        for j in range(i + 1, n_rows):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    x = np.zeros((n_rows, 1))
    for i in range(n_rows - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n_rows], x[i+1:])

    return x

try:
    filas = int(input("Ingrese el número de filas de la matriz A: "))
    columnas = int(input("Ingrese el número de columnas de la matriz A: "))
except ValueError:
    print("Error: debe ingresar un número entero para el tamaño de la matriz.")
    exit()

A = np.zeros((filas, columnas))
B = np.zeros((filas, 1))

print(f"Ingrese los coeficientes de la matriz A ({filas}x{columnas}):")
for i in range(filas):
    print("*" * 20)
    for j in range(columnas):
        while True:
            try:
                A[i][j] = float(input(f"Ingrese el coeficiente A[{i+1},{j+1}]: ") or 0)
                break
            except ValueError:
                print("Error: debe ingresar un número válido.")

print("*" * 20)
print(f"Ingrese los términos independientes de la matriz B ({filas}x1):")
for i in range(filas):
    while True:
        try:
            B[i][0] = float(input(f"Ingrese el término B[{i+1},1]: ") or 0)
            break
        except ValueError:
            print("Error: debe ingresar un número válido.")

try:
    solucion = gaussian_elimination(A, B)
    print("La solución del sistema de ecuaciones es:")
    print(solucion)
except ValueError as e:
    print("Error:", e)
except Exception as e:
    print("Error inesperado:", e)
