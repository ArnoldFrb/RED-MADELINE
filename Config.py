import random as rn
import numpy as np
import pandas as pd
import os
from Layers import *
from heapq import nsmallest
import tkinter as tk


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
    def Entrenar(self, rataAprendizaje, errorMaximo, numeroIteraciones, funcionSalida):

        layers = Layers()
        self.Entradas = self.NormalizarEntradas(self.Entradas)
        self.Salidas = self.NormalizarSalidas(self.Salidas) if len(self.Salidas)==1 else self._NormalizarSalidas(self.Salidas)

        DimensionPesos = []
        DimensionUmbrales = []
        for I in range(len(self.capas)):

            # CONDICION PARA ENTRADA Y CAPA 1
            if(I == 0):
                DimensionPesos.append(self.Generar_pesos(len(self.Entradas[0]), self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
             # CONDICION PARA CAPAS INTERMEDIAS
            if(I > 0 & I < (len(self.capas) - 1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I-1][1], self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
            # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
            if(I >= (len(self.capas)-1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I][1], len(self.Salidas[0])))
                DimensionUmbrales.append(self.Generar_Umbrales(len(self.Salidas[0])))
        
        for x in range(len(self.Entradas)): # ARRAY PARA PRESENTAR PATRONES

            entrada = self.Entradas[x][:]
            salida = self.Salidas[x][:]

            print('//////////////////')
            print('////////////////////////////////////')
            print('PATRON', x+1)
            print()
            
            NuevaEntrada = []

            for y in range(len(self.capas)): # ARRAY PARA RECORER CAPAS

                if(y == 0):
                    print('ENTRADAS x CAPA', y,'=>' , len(entrada), 'x', self.capas[y][1])

                    print('ENTRADAS')
                    print(np.array(entrada))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[y][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[y][:][:]))
                    print()

                    print('FUNCION ACTIVACION NO LINEAL:', self.capas[y][2])
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(self.capas[y][2]), 
                            layers.FuncionSoma(entrada, DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
                    print(np.array(NuevaEntrada))
                    print()
                    print('----------------------------------------')
                    print()

                if(y > 0 & y < (len(self.capas) - 1)):
                    print('CAPA', y-1, 'x CAPA', y,'=>' , self.capas[y-1][1], 'x', self.capas[y][1])
                    
                    print('ENTRADAS')
                    print(np.array(NuevaEntrada[y-1]))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[y][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[y][:][:]))
                    print()

                    print('FUNCION ACTIVACION NO LINEAL:', self.capas[y][2])
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(self.capas[y][2]), 
                            layers.FuncionSoma(NuevaEntrada[y-1], DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
                    print(np.array(NuevaEntrada[y]))
                    print()
                    print('----------------------------------------')
                    print()

                if(y >= (len(self.capas)-1)):
                    print('CAPA', y, 'x SALIDAS','=>' , self.capas[y][1], 'x', len(salida))
                    
                    print('ENTRADAS')
                    print(np.array(NuevaEntrada[y]))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[len(self.capas)][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[len(self.capas)][:][:]))
                    print()

                    print('FUNCION ACTIVACION LIENAL:', funcionSalida)
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(funcionSalida), 
                            layers.FuncionSoma(NuevaEntrada[y], DimensionPesos[len(self.capas)][:][:], DimensionUmbrales[len(self.capas)][:][:])))
                    print(np.array(NuevaEntrada[y+1]))
                    print()

                    print('ERROR LINEAL')
                    ErrorLineal = layers.ErrorLineal(salida, NuevaEntrada[y+1])
                    print(ErrorLineal)
                    print()

                    print('ERROR PATRON')
                    _ErrorPatron = sum(ErrorLineal) / len(ErrorLineal)
                    print(_ErrorPatron)
                    print()
                    
                    CapasInversa = self.capas[::-1]
                    _DimensionPesos = DimensionPesos[::-1]
                    _DimensionUmbrales = DimensionUmbrales[::-1]
                    NuevaEntrada_ = NuevaEntrada[::-1]

                    _PesosTempo = 0
                    _UmbralesTempo = 0

                    print('///////////////////////////////////////')
                    print('------ACTUALIZAR PESOS Y UMBRALES------')
                    print()

                    for z in range(len(CapasInversa)):

                        if(z == 0):
                            ErrorLienalMenor = np.amin(ErrorLineal)
                            indexM = ErrorLineal.index(ErrorLienalMenor)
                            print('ERROR LINEAL MENOR:', ErrorLienalMenor, 'INDEX:', indexM)
                            print()

                            print('ENTRADAS:')
                            print(np.array(NuevaEntrada_[z]))
                            print()

                            print('PESOS TEMPORALES')
                            PesosTemporales = layers.ActualizarPesosSalidas(
                                _DimensionPesos[z][:][:], indexM, int(rataAprendizaje), ErrorLineal, NuevaEntrada_[z])
                            print(np.array(PesosTemporales))
                            print()

                            print('UMBRALES TEMPORALES')
                            UmbralesTemporales = layers.ActualizarUmbralesSalidas(
                                _DimensionUmbrales[z][:][:], indexM, int(rataAprendizaje), ErrorLineal)
                            print(np.array(UmbralesTemporales))
                            print()

                            print('NUEVO FUNCION ACTIVACION LIENAL:', funcionSalida)
                            _NuevaEntrada = layers.FuncionActivacionCapas(
                                layers._FuncionActivacion(funcionSalida), 
                                layers.FuncionSoma(NuevaEntrada_[z+1], PesosTemporales, UmbralesTemporales))
                            print(np.array(_NuevaEntrada))
                            print()

                            print('NUEVO ERROR LINEAL')
                            _ErrorLineal = layers.ErrorLineal(salida, _NuevaEntrada)
                            print(_ErrorLineal)
                            print()

                            print('ES MENOR EL NUEVO ERROR:', _ErrorLineal[indexM] < ErrorLineal[indexM])
                            print()

                            if(_ErrorLineal[indexM] < ErrorLineal[indexM]):
                                _DimensionPesos[z][:][:] = PesosTemporales
                                _DimensionUmbrales[z][:][:] = UmbralesTemporales
                                print('PESOS ACTUALIZADOS')
                                print(np.array(_DimensionPesos[z][:][:]))
                                print()

                                print('UMBRALES ACTUALIZADOS')
                                print(np.array(_DimensionUmbrales[z][:][:]))
                                print()
                            else:
                                print('NO SE ACTUALIZAN PESOS Y UMBRALES')
                                print('PESOS')
                                print(np.array(_DimensionPesos[z][:][:]))
                                print()

                                print('UMBRALES')
                                print(np.array(_DimensionUmbrales[z][:][:]))
                                print()

                            print('//////////////////////////////////////////////////////////////////')
                            print('//////////////////////////////////////////////////////////////////')
                            print()

                    DimensionPesos = _DimensionPesos[::-1]
                    DimensionUmbrales = _DimensionUmbrales[::-1]

        print()
        print('//////////// - FIN ENTRENAMIENTO - ////////////')
        print()

    # LIMPIAR CAPAS
    def Limpiar(self):
        self.capas = []

    def NormalizarEntradas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            aux = []
            for j in range(len(salida)):
                aux.append(salida[j][i])
            salidas.append(aux)
        return salidas

    def NormalizarSalidas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            s = []
            for j in range(len(salida)):
                s.append(salida[j][i])
            salidas.append(s)
        return salidas

    def _NormalizarSalidas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            aux = []
            for j in range(len(salida)):
                aux.append(salida[j][i])
            salidas.append(aux)
        return salidas
