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
    def Generar_pesos(self, row, col):
        Matriz = []
        for N in range(row):
            Fila = []
            for M in range(col):
                Fila.append(round(rn.uniform(-1, 1), 2))
            Matriz.append(Fila)
        return Matriz

    #METODO PARA GENERAR UMBRALES
    def Generar_Umbrales(self, row):
        Fila = []
        for N in range(row):
            Fila.append(round(rn.uniform(-1, 1), 2))
        return Fila

    # LLENAR MATRICES ENTRADAS Y SALIDAS
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

    # AGREGAR CAPAS OCULTAS
    def AgregarCapas(self, capa, neuronas, funcActivacion):
        encabezado = ['Capa', 'Neuronas', 'Func Activacion']
        self.capas.append([capa, neuronas, funcActivacion])
        
        return pd.DataFrame(data=self.capas, columns=encabezado)

    # INICIAR ENTRENAMIENTO
    def Entrenar(self, rataAprendizaje, errorLineal, numeroIteraciones):
        print("OK")

    # LIMPIAR CAPAS
    def Limpiar(self):
        self.capas = []

if __name__ == '__main__':
    print("Hola") 