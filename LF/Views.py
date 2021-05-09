import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
from Config import *

class Views:

    def __init__(self, window):
        self.wind = window
        self.wind.title("RED MADELINE")
        self.wind.resizable(0,0)
        self.wind.geometry("1100x600")
        self.wind.winfo_screenheight()
        self.wind.winfo_screenwidth()
        self.config = Config()

        # FRAME PRINCIPAL
        frameMain = tk.Frame(master=self.wind, width=1100, height=600, background="#e3e3e3")
        frameMain.place(relx=.0, rely=.0)

        # FRAME PARQ CONFIGURAR LA RATA DE APREN., NUMERO DE ITERA. Y ERROR MAXIMO
        self.frameConfig = tk.Frame(frameMain, width=450, height=50, background="#fafafa")
        self.frameConfig.place(relx=.01, rely=.02)

        tk.Label(self.frameConfig, text="PARAMETROS DE ENTRENAMIENTO", bg="#fafafa").place(relx=.01, rely=.001)

        btnData = tk.Button(self.frameConfig, text="Cargar Data", command= self.Event_btnData,
         relief="flat", overrelief="flat", bg="#e3e3e3", borderwidth=2)
        btnData.place(relx=.01, rely=.4)

        tk.Label(self.frameConfig, text="RATA:", bg="#fafafa").place(relx=.218, rely=.5)
        self.entRataAprendizaje = tk.Entry(self.frameConfig, width=5)
        self.entRataAprendizaje.place(relx=.3, rely=.5)
        self.entRataAprendizaje.insert(0, 1)

        tk.Label(self.frameConfig, text="ERROR:", bg="#fafafa").place(relx=.403, rely=.5)
        self.entErrorMaximo = tk.Entry(self.frameConfig, width=5)
        self.entErrorMaximo.place(relx=.5, rely=.5)
        self.entErrorMaximo.insert(0,0.001)

        tk.Label(self.frameConfig, text="ITERA:", bg="#fafafa").place(relx=.615, rely=.5)
        self.entNumeroIteraciones = tk.Entry(self.frameConfig, width=9)
        self.entNumeroIteraciones.place(relx=.7, rely=.5)
        self.entNumeroIteraciones.insert(0, 10000)

        # FRAME PARA VISUALIZAR ENTRADAS, SALIDAS Y PATRONES
        self.frameConfigInicial = tk.Frame(frameMain, width=450, height=60, background="#fafafa")
        self.frameConfigInicial.place(relx=.01, rely=.115)

        tk.Label(self.frameConfigInicial, text="CONFIG ENTRENAMIENTO", bg="#fafafa").place(relx=.34, rely=.01)
        tk.Label(self.frameConfigInicial, text="ENTRADAS", bg="#fafafa").place(relx=.1, rely=.3)
        tk.Label(self.frameConfigInicial, text="SALIDAS", bg="#fafafa").place(relx=.45, rely=.3)
        tk.Label(self.frameConfigInicial, text="PATRONES", bg="#fafafa").place(relx=.8, rely=.3)

        # FRAME PARA VISUALIZAR LOS DATOS DE ENTRENAMIENTO
        self.frameData = tk.Frame(frameMain, background="#fafafa", width=450, height=264)
        self.frameData.place(relx=.01, rely=.227)

        # FRAME PARA CONFIGURAR Y VISUALIZAR LAS CAPAS Y FUNCIONES DE ACTIVACION
        self.frameConfigCapas = tk.Frame(frameMain, width=450, height=180, background="#fafafa")
        self.frameConfigCapas.place(relx=.01, rely=.678)

        tk.Label(self.frameConfigCapas, text="CONFIGURAR CAPAS", bg="#fafafa").place(relx=.02, rely=.01)

        self.btnAgregar = tk.Button(self.frameConfigCapas, text="Agregar", state=tk.DISABLED, command= self.Event_btnAgregarCapas,
         relief="flat", overrelief="flat", bg="#e3e3e3", borderwidth=2)
        self.btnAgregar.place(relx=.875, rely=.01)

        self.btnLimpiar = tk.Button(self.frameConfigCapas, text="Limpiar", state=tk.DISABLED, command= self.Event_btnLimpiar,
         relief="flat", overrelief="flat", bg="#e3e3e3", borderwidth=2)
        self.btnLimpiar.place(relx=.755, rely=.01)
        
        tk.Label(self.frameConfigCapas, text="CAPA:", bg="#fafafa").place(relx=.003, rely=.170)
        self.entCapa = tk.Entry(self.frameConfigCapas, width=5)
        self.entCapa.place(relx=.1, rely=.170)
        self.capa = 1
        self.entCapa.insert(0, self.capa)

        tk.Label(self.frameConfigCapas, text="NEURONAS:", bg="#fafafa").place(relx=.20, rely=.170)
        self.entNeuronas = tk.Entry(self.frameConfigCapas, width=5)
        self.entNeuronas.place(relx=.357, rely=.170)
        self.entNeuronas.insert(0, 10)

        tk.Label(self.frameConfigCapas, text="FUNC ACTIVACION:", bg="#fafafa").place(relx=.43, rely=.170)
        self.comboBoxCapasOcultas = ttk.Combobox(self.frameConfigCapas, state=tk.DISABLED)
        self.comboBoxCapasOcultas["values"] = ["SIGMOIDE", "TANGENTE H.", "GAUSSIANA"]
        self.comboBoxCapasOcultas.place(relx=.68, rely=.170)

        # FRAME PARQA VISUALIZAR LA CONFIGURACION DE LAS CAPAS
        self.frameConfigCapasData = tk.Frame(self.frameConfigCapas, width=265, height=122, background="#fafafa")
        self.frameConfigCapasData.place(relx=0, rely=.32)

        tk.Label(self.frameConfigCapas, text="FUNCIONES DE ACTIVACION", bg="#fafafa").place(relx=.615, rely=.5)
        tk.Label(self.frameConfigCapas, text="CAPA SALIDA", bg="#fafafa").place(relx=.615, rely=.6)
        self.comboBoxCapaSalida = ttk.Combobox(self.frameConfigCapas, state=tk.DISABLED)
        self.comboBoxCapaSalida["values"] = ["SIGMOIDE", "ESCALON", "LINEAL"]
        self.comboBoxCapaSalida.place(relx=.62, rely=.7)

        self.frameEntrenar = tk.Frame(frameMain, width=620, height=50, background="#fafafa")
        self.frameEntrenar.place(relx=.426, rely=.02)

        self.btnEntrenar = tk.Button(self.frameEntrenar, text="Entrenar", state=tk.DISABLED, command= self.Event_btnEntrenar,
         relief="flat", overrelief="flat", bg="#e3e3e3", borderwidth=2)
        self.btnEntrenar.place(relx=.01, rely=.4)

        self.frameEntranamiento = tk.Frame(frameMain, width=620, height=228, background="#fafafa")
        self.frameEntranamiento.place(relx=.426, rely=.115)

        self.frameSimulacion = tk.Frame(frameMain, width=620, height=283, background="#fafafa")
        self.frameSimulacion.place(relx=.426, rely=.506)

    def Event_btnData(self):
        
        self.ruta = filedialog.askopenfilename()
        self.config.NormalizarDatos(self.ruta)
        Matriz = pd.read_csv(self.ruta, delimiter=' ')

        treeView = ttk.Treeview(self.frameData)
        self.CrearGrid(treeView, self.frameData)
        self.LlenarTabla(treeView, Matriz)

        tk.Label(self.frameConfigInicial, text=str(len(self.config.Entradas)), bg="#fafafa").place(relx=.15, rely=.6)
        tk.Label(self.frameConfigInicial, text=str(len(self.config.Salidas)), bg="#fafafa").place(relx=.49, rely=.6)
        tk.Label(self.frameConfigInicial, text=str(len(self.config.Entradas[0])), bg="#fafafa").place(relx=.85, rely=.6)

        self.btnAgregar['state'] = tk.NORMAL
        self.comboBoxCapasOcultas['state'] = tk.NORMAL
        self.comboBoxCapaSalida['state'] = tk.NORMAL

    def Event_btnEntrenar(self):

        if(self.comboBoxCapaSalida.get() == ''):
            messagebox.showinfo(message="No ha seleccinado una funcion de activacion para las salidas", title="ERROR")
            return

        self.config.Entrenar(self.entRataAprendizaje.get(), self.entErrorMaximo.get(), self.entNumeroIteraciones.get())

    def Event_btnAgregarCapas(self):

        if(self.comboBoxCapasOcultas.get() == ''):
            messagebox.showinfo(message="No ha seleccinado una funcion de activacion para la capa oculta", title="ERROR")
            return

        treeView = ttk.Treeview(self.frameConfigCapasData)
        self.CrearGrid(treeView, self.frameConfigCapasData)
        self.LlenarTabla(treeView, self.config.AgregarCapas(int(self.entCapa.get()), int(self.entNeuronas.get()), self.comboBoxCapasOcultas.get()))

        self.btnEntrenar['state'] = tk.NORMAL
        self.btnLimpiar['state'] = tk.NORMAL

        self.capa = self.capa + 1
        self.entCapa.delete(0, tk.END)
        self.entCapa.insert(0, self.capa)

    def LlenarTabla(self, treeView, Matriz):
        treeView.delete(*treeView.get_children())
        treeView["column"] = list(Matriz.columns)
        treeView["show"] = "headings"

        for column in treeView["columns"]:
            treeView.column(column=column, width=100)
            treeView.heading(column=column, text=column)

        Matriz_rows1 = Matriz.to_numpy().tolist()
        for row in Matriz_rows1:
            treeView.insert("", "end", values=row)

    def CrearGrid(self, treeView, frame):
        style = ttk.Style(frame)
        style.configure(treeView, rowheight=100, highlightthickness=0, bd=0)  
        treeView.place(relheight=1, relwidth=1)

    def Event_btnLimpiar(self):
        self.config.Limpiar()
        treeView = ttk.Treeview(self.frameConfigCapasData)
        self.CrearGrid(treeView, self.frameConfigCapasData)

    def FuncionesSalidas(self, e):
        if ("SIGMOIDE" == e):
            return 1

        if ("ESCALON" == e):
            return 2

        if ("LINEAL" == e):
            return 3

        if ("TANGENTE H." == e):
            return 2

        if ("GAUSSIANA" == e):
            return 3

if __name__ == '__main__':
    winw = tk.Tk()
    Views(winw)
    winw.mainloop()