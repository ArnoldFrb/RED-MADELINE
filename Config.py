import random as rn
import numpy as np
import pandas as pd
import os

class Config:

    #CONSTRUCTOR
    def __init__(self):
        self.Entradas = []
        self.Salidas = []
        self.Entranamiento = ''
        self.capas = []
    
    #METODO PARA GENERAR PESOS
    def Generar_pesos(self):
        Matriz = []
        for N in range(len(self.Salidas)):
            Fila = []
            for M in range(len(self.Entradas)):
                Fila.append(round(rn.uniform(-1, 1), 2))
            Matriz.append(Fila)
        return Matriz

    #METODO PARA GENERAR UMBRALES
    def Generar_Umbrales(self):
        Fila = []
        for N in range(len(self.Salidas)):
            Fila.append(round(rn.uniform(-1, 1), 2))
        return Fila

    def NormalizarDatos(self, ruta):
        Matriz = pd.read_csv(ruta, delimiter=' ')
        col = Matriz.columns
        column = Matriz.to_numpy()
        self.Entranamiento = os.path.basename(os.path.splitext(ruta)[0])

        for i in range(len(col)):
            if 'X' in col[i]:
                Fila = []
                for j in range(len(column)):
                    Fila.append(column[j,i])
                self.Entradas.append(Fila)
            else:
                Fila = []
                for j in range(len(column)):
                    Fila.append(column[j,i])
                self.Salidas.append(Fila)

    def InicializarCapas(self, numeroCapas, rangoA, rangoB):
        encabezado = ['Capa', 'Neuronas']
        for I in range(numeroCapas):
            filas = []
            filas.append(I+1)
            filas.append(round(rn.uniform(rangoA, rangoB)))
            self.capas.append(filas)

        return pd.DataFrame(data=self.capas, columns=encabezado)

    def Entrenar(self, rataAprendizaje, errorLineal, numeroIteraciones):
        print("OK")

if __name__ == '__main__':
    print("Hola") 