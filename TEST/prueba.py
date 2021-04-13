import random as rn

def InicializarCapas(numeroCapas, rangoA, rangoB):
    capas = []
    for I in range(numeroCapas):
        filas = []
        filas.append(I+1)
        filas.append(round(rn.uniform(rangoA, rangoB)))
        capas.append(filas)

    print(capas)

InicializarCapas(10, 10, 15)