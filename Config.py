import random as rn
import numpy as np
import pandas as pd
import os
from Funtions import *
from heapq import nsmallest
import tkinter as tk
import copy as cp

class Config:

    #CONSTRUCTOR
    def __init__(self):
        self.Entradas = []
        self.Salidas = []
        self.Entranamiento = ''
        self.capas = []

    # LLENAR MATRICES ENTRADAS Y SALIDAS
    def NormalizarDatos(self, ruta):
        self.Entradas = []
        self.Salidas = []

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

        func = Funtions()
        self.Entradas = self.NormalizarEntradas(self.Entradas)
        self.Salidas = self.NormalizarSalidas(self.Salidas) if len(self.Salidas)==1 else self._NormalizarSalidas(self.Salidas)

        DimensionPesos = []
        DimensionUmbrales = []
        for I in range(len(self.capas)):

            # CONDICION PARA ENTRADA Y CAPA 1
            if(I == 0):
                DimensionPesos.append(func.Generar_pesos(len(self.Entradas[0]), self.capas[I][1]))
                DimensionUmbrales.append(func.Generar_Umbrales(self.capas[I][1]))
                
             # CONDICION PARA CAPAS INTERMEDIAS
            if(I > 0 & I < (len(self.capas) - 1)):
                DimensionPesos.append(func.Generar_pesos(self.capas[I-1][1], self.capas[I][1]))
                DimensionUmbrales.append(func.Generar_Umbrales(self.capas[I][1]))
                
            # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
            if(I >= (len(self.capas)-1)):
                DimensionPesos.append(func.Generar_pesos(self.capas[I][1], len(self.Salidas[0])))
                DimensionUmbrales.append(func.Generar_Umbrales(len(self.Salidas[0])))
        
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
                        func.FuncionActivacionCapas(
                            func._FuncionActivacion(self.capas[y][2]), 
                            func.FuncionSoma(entrada, DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
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
                        func.FuncionActivacionCapas(
                            func._FuncionActivacion(self.capas[y][2]), 
                            func.FuncionSoma(NuevaEntrada[y-1], DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
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
                        func.FuncionActivacionCapas(
                            func._FuncionActivacion(funcionSalida), 
                            func.FuncionSoma(NuevaEntrada[y], DimensionPesos[len(self.capas)][:][:], DimensionUmbrales[len(self.capas)][:][:])))
                    print(np.array(NuevaEntrada[y+1]))
                    print()

                    print('ERROR LINEAL')
                    ErrorLineal = func.ErrorLineal(salida, NuevaEntrada[y+1])
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

                    print('///////////////////////////////////////')
                    print('------ACTUALIZAR PESOS Y UMBRALES------')
                    print()

                    self.Error_Lineal = []

                    for z in range(len(CapasInversa)):

                        if(z == 0):
                            print('SALIDA')
                            print()

                            ErrorLienalMenor = np.amin(ErrorLineal)
                            indexM = ErrorLineal.index(ErrorLienalMenor)
                            print('ERROR LINEAL MENOR:', ErrorLienalMenor, 'INDEX:', indexM)
                            print()

                            print('ENTRADAS:')
                            print(np.array(NuevaEntrada_[z]))
                            print()

                            print('PESOS TEMPORALES')
                            PesosTemporales = func.ActualizarPesosSalidas(
                                cp.deepcopy(_DimensionPesos[z]), indexM, int(rataAprendizaje), ErrorLineal, NuevaEntrada_[z])
                            print(np.array(PesosTemporales))
                            print()

                            print('UMBRALES TEMPORALES')
                            UmbralesTemporales = func.ActualizarUmbralesSalidas(
                                cp.deepcopy(_DimensionUmbrales[z]), indexM, int(rataAprendizaje), ErrorLineal)
                            print(np.array(UmbralesTemporales))
                            print()

                            print('NUEVO FUNCION ACTIVACION LIENAL:', funcionSalida)
                            _NuevaEntrada = func.FuncionActivacionCapas(
                                func._FuncionActivacion(funcionSalida), 
                                func.FuncionSoma(NuevaEntrada_[z+1], PesosTemporales, UmbralesTemporales))
                            print(np.array(_NuevaEntrada))
                            print()

                            print('NUEVO ERROR LINEAL')
                            self.Error_Lineal.append(func.ErrorLineal(salida, _NuevaEntrada))
                            print(self.Error_Lineal[z])
                            print()

                            print('ES MENOR EL NUEVO ERROR:', self.Error_Lineal[z][indexM] < ErrorLineal[indexM])
                            print('NUEVO:', self.Error_Lineal[z][indexM], '< ANTIGUAO:', ErrorLineal[indexM])
                            print()

                            if(self.Error_Lineal[z][indexM] < ErrorLineal[indexM]):
                                _DimensionPesos[z] = cp.deepcopy(PesosTemporales)
                                _DimensionUmbrales[z] = cp.deepcopy(UmbralesTemporales)
                                print('PESOS ACTUALIZADOS')
                                print(np.array(_DimensionPesos[z]))
                                print()

                                print('UMBRALES ACTUALIZADOS')
                                print(np.array(_DimensionUmbrales[z]))
                                print()
                            else:
                                print('NO SE ACTUALIZAN PESOS Y UMBRALES')
                                print('PESOS')
                                print(np.array(_DimensionPesos[z]))
                                print()

                                print('UMBRALES')
                                print(np.array(_DimensionUmbrales[z]))
                                print()

                            print('//////////////////////////////////////////////////////////////////')
                            print('//////////////////////////////////////////////////////////////////')
                            print()
                        
                        if(z > 0 & z < (len(CapasInversa) - 1)):

                            self.Error_Lineal.append(func.ErrorNoLineal(self.Error_Lineal[z-1], _DimensionPesos[z-1][:][:]))
                            print('ERROR NO LIEAL CAPA', z)
                            print(np.array(self.Error_Lineal[z]))
                            print()

                            print('PESOS:')
                            print(np.array(_DimensionPesos[z]))
                            print(id(_DimensionPesos[z]))
                            print()

                            print('Umbrales:')
                            print(np.array(_DimensionUmbrales[z]))
                            print()

                            _pesos = cp.deepcopy(_DimensionPesos)
                            _umbrales = cp.deepcopy(_DimensionUmbrales)

                            _Error_Lineal = cp.deepcopy(self.Error_Lineal[z])

                            for i in range(len(_Error_Lineal)):
                                if(_Error_Lineal):
                                    Cero = nsmallest(i+1, _Error_Lineal, key=lambda x: abs(x-0))
                                    print('ERRORES CERCANO A CERO:', i+1)
                                    print(np.array(Cero))
                                    print()

                                    for error in Cero:
                                        index = self.Error_Lineal[z].index(error)
                                        print('ERROR LINEAL CERCANO A CERO:', error, 'INDEX:', index)
                                        print()

                                        func.ActualizarPesosCapas(
                                            _pesos[z], index, int(rataAprendizaje), _ErrorPatron, NuevaEntrada_[z]
                                        )

                                        print('PESOS TEMPORALES:')
                                        print(np.array(_pesos[z]))
                                        print()

                                        func.ActualizarUmbralesCapas(
                                            _umbrales[z], index, int(rataAprendizaje), _ErrorPatron
                                        )

                                        print('UMBRALES TEMPORALES:')
                                        print(np.array(_umbrales[z]))
                                        print()

                                        _pesos = _pesos[::-1]
                                        _umbrales = _umbrales[::-1]

                                        ind = 0
                                        for j in range(len(self.capas)):
                                            if (_DimensionPesos[z] == DimensionPesos[j]):
                                                ind = j

                                        for j in range(ind, len(self.capas)):
                                            
                                            if(j > 0 & j < (len(self.capas) - 1)):
                                                
                                                NuevaEntrada[j] = (
                                                    func.FuncionActivacionCapas(
                                                        func._FuncionActivacion(self.capas[j][2]), 
                                                        func.FuncionSoma(NuevaEntrada[j-1], _pesos[j], _umbrales[j])))

                                            if(j >= (len(self.capas)-1)):
                                                
                                                NuevaEntrada[j+1] = (
                                                    func.FuncionActivacionCapas(
                                                        func._FuncionActivacion(funcionSalida), 
                                                        func.FuncionSoma(NuevaEntrada[j], _pesos[len(self.capas)], _umbrales[len(self.capas)])))
                                                
                                                NuevoErrorLineal = func.ErrorLineal(salida, cp.deepcopy(NuevaEntrada[j+1]))

                                                print('ES MENOR EL NUEVO ERROR:', NuevoErrorLineal < ErrorLineal)
                                                print('NUEVO:', NuevoErrorLineal, '< ANTIGUAO:', ErrorLineal)
                                                print()

                                                if(NuevoErrorLineal < ErrorLineal):
                                                    _pesos = _pesos[::-1]
                                                    _umbrales = _umbrales[::-1]
                                                    _DimensionPesos = cp.deepcopy(_pesos)
                                                    _DimensionUmbrales = cp.deepcopy(_umbrales)

                                                    print('PESOS ACTUALIZADOS')
                                                    print(np.array(_DimensionPesos[z]))
                                                    print()

                                                    print('UMBRALES ACTUALIZADOS')
                                                    print(np.array(_DimensionUmbrales[z]))
                                                    print()
                                                else:
                                                    print('NO SE ACTUALIZAN PESOS Y UMBRALES')
                                                    print('PESOS')
                                                    print(np.array(_DimensionPesos[z]))
                                                    print()

                                                    print('UMBRALES')
                                                    print(np.array(_DimensionUmbrales[z]))
                                                    print()

                                        _Error_Lineal = list( filter(lambda x: x != error, _Error_Lineal) )

                                    print('-------------------------------------------------------------')
                                    print()
                                    print()

                        if(z >= (len(CapasInversa)-1)):

                            self.Error_Lineal.append(func.ErrorNoLineal(self.Error_Lineal[z], _DimensionPesos[z][:][:]))
                            print('ERROR NO LIEAL CAPA', z+1)
                            print(np.array(self.Error_Lineal[z+1]))
                            print()

                            print('PESOS:')
                            print(np.array(_DimensionPesos[z+1]))
                            print(id(_DimensionPesos[z+1]))
                            print()

                            print('Umbrales:')
                            print(np.array(_DimensionUmbrales[z+1]))
                            print()

                            _pesos = cp.deepcopy(_DimensionPesos)
                            _umbrales = cp.deepcopy(_DimensionUmbrales)

                            _Error_Lineal = cp.deepcopy(self.Error_Lineal[z+1])

                            for i in range(len(_Error_Lineal)):
                                if(_Error_Lineal):
                                    Cero = nsmallest(i+1, _Error_Lineal, key=lambda x: abs(x-0))
                                    print('ERRORES CERCANO A CERO:', i+1)
                                    print(np.array(Cero))
                                    print()

                                    for error in Cero:
                                        index = self.Error_Lineal[z+1].index(error)
                                        print('ERROR LINEAL CERCANO A CERO:', error, 'INDEX:', index)
                                        print()

                                        func.ActualizarPesosCapas(
                                            _pesos[z+1], index, int(rataAprendizaje), _ErrorPatron, NuevaEntrada_[z+1]
                                        )

                                        print('PESOS TEMPORALES:')
                                        print(np.array(_pesos[z+1]))
                                        print()

                                        func.ActualizarUmbralesCapas(
                                            _umbrales[z+1], index, int(rataAprendizaje), _ErrorPatron
                                        )

                                        print('UMBRALES TEMPORALES:')
                                        print(np.array(_umbrales[z+1]))
                                        print()

                                        _pesos = _pesos[::-1]
                                        _umbrales = _umbrales[::-1]

                                        ind = 0
                                        for j in range(len(self.capas)):
                                            if (_DimensionPesos[z+1] == DimensionPesos[j]):
                                                ind = j

                                        for j in range(ind, len(self.capas)):
                                            
                                            if(j > 0 & j < (len(self.capas) - 1)):
                                                
                                                NuevaEntrada[j] = (
                                                    func.FuncionActivacionCapas(
                                                        func._FuncionActivacion(self.capas[j][2]), 
                                                        func.FuncionSoma(NuevaEntrada[j-1], _pesos[j], _umbrales[j])))

                                            if(j >= (len(self.capas)-1)):
                                                
                                                NuevaEntrada[j+1] = (
                                                    func.FuncionActivacionCapas(
                                                        func._FuncionActivacion(funcionSalida), 
                                                        func.FuncionSoma(NuevaEntrada[j], _pesos[len(self.capas)], _umbrales[len(self.capas)])))
                                                
                                                NuevoErrorLineal = func.ErrorLineal(salida, cp.deepcopy(NuevaEntrada[j+1]))

                                                print('ES MENOR EL NUEVO ERROR:', NuevoErrorLineal < ErrorLineal)
                                                print('NUEVO:', NuevoErrorLineal, '< ANTIGUAO:', ErrorLineal)
                                                print()

                                                if(NuevoErrorLineal < ErrorLineal):
                                                    _pesos = _pesos[::-1]
                                                    _umbrales = _umbrales[::-1]
                                                    _DimensionPesos = cp.deepcopy(_pesos)
                                                    _DimensionUmbrales = cp.deepcopy(_umbrales)

                                                    print('PESOS ACTUALIZADOS')
                                                    print(np.array(_DimensionPesos[z+1]))
                                                    print()

                                                    print('UMBRALES ACTUALIZADOS')
                                                    print(np.array(_DimensionUmbrales[z+1]))
                                                    print()
                                                else:
                                                    print('NO SE ACTUALIZAN PESOS Y UMBRALES')
                                                    print('PESOS')
                                                    print(np.array(_DimensionPesos[z+1]))
                                                    print()

                                                    print('UMBRALES')
                                                    print(np.array(_DimensionUmbrales[z+1]))
                                                    print()

                                        _Error_Lineal = list( filter(lambda x: x != error, _Error_Lineal) )

                                    print('//////////// - FIN PATRON - ////////////')
                                    print()
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
