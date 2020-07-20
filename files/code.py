# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: FEDERICO RAFAEL GARCIA GARCIA
"""

import numpy as np
import math
import matplotlib as mplt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.svm import SVR

# Cargamos los datos
def CargarDatos(file, tipo, separador):
    # Abrimos el fichero en modo lectura
    f = open(file, "r")
    
    # Cada linea
    lineas = f.readlines()

    
    # Por cada linea, obtenemos datos
    datos = []
    for l in lineas:
        # Separar por coma
        datos_linea = l.rstrip().split(separador)
        
        # Anadir convertidos a enteros
        if tipo == 'int':
            datos.append(list(map(int, datos_linea)))
        elif tipo == 'float':
            datos.append(list(map(float, datos_linea)))

    # Numero de valores
    n = len(datos_linea)

    # La etiqueta/output va al final. Separar en X e Y y en formato numpy
    x = np.array([x[:n-1] for x in datos]) # Todos los valores menos ultimo
    y = np.array([x[n-1] for x in datos]) # Ultimo valor

    return x, y

# Muestra 10x10 digitos
def MostrarNumeros(x, labels):
    # Separar por digitos
    data = np.array([None] * 10)
    for n in range(10):
        data[n] = x[np.nonzero(labels == n)]
        
    # Matriz de destino
    matriz = np.zeros((80, 80), dtype='uint8')
    
    for n in range(100):
        i = int(n/10)
        j = n%10

        digito = np.reshape(data[i][j], (8, 8)) # Redimsensionar de 64 a 8x8

        #Copiar a la matriz
        matriz[i*8:(i+1)*8,j*8:(j+1)*8] = digito

    # Mostrar matriz
    plt.axis('off')
    plt.imshow(matriz, cmap='gray', vmin=0, vmax=16, interpolation='nearest')
    plt.show()
    
    
    pass

# Convertimos el array 64 de un numero a [8, 8] y devolvemos un array con
# tres caractersticas: simetria vertical, intensidad superior e inferior
def ManuscritoA3D(x):
    
    # Redimsensionar de 64 a 8x8
    data = np.reshape(x, (8, 8))
    
    # Simetria vertical
    sim_ver = np.sum(np.abs(data[0:4,:]-data[4:8,:]))
    
    # Intensidad superior e inferior
    int_sup = np.sum(data[0:4,:])
    int_inf = np.sum(data[4:8,:])

    return np.array([sim_ver, int_sup, int_inf])

# Convertir el conjunto de elementos a caracteristicas 3D
def DataA3D(x):
    x3D = np.array([None]*len(x))
    for i in range(len(x)):
        x3D[i] = ManuscritoA3D(x[i])
    
    return x3D

# Transformamos el array 64 de un numero a [8, 8]
def ManuscritoA16D(x):
    
    # Redimsensionar de 64 a 8x8
    data = np.reshape(x, (8, 8))

    caracteristicas = np.array([0.0] * 16)

    # Intensidades fila
    for i in range(8):
        caracteristicas[i]   = np.sum(data[i,:])
        caracteristicas[8+i] = np.sum(data[:,i])
        
    return caracteristicas

# Convertir el conjunto de elementos a caracteristicas 16D
def DataA16D(x):
    x16D = np.array([None]*len(x))
    for i in range(len(x)):
        x16D[i] = ManuscritoA16D(x[i])
    
    return x16D

# Muestra datos en 2D de una lista de datos, pudiendo ordenarlos segun el primero
def Plot2Ds(ys, xlabel, ylabel, sort='Si'):
    fig = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if(sort=='Si'):
        indices = np.argsort(ys[0])
        
    xrange = np.arange(0, len(ys[0]))
    
    for i in range(len(ys)):
        if(sort=='Si'):
            y_sort = [ys[i][j] for j in indices]
            plt.scatter(xrange, y_sort, s=10)
        else:
            plt.scatter(xrange, ys[i], s=10)

    plt.show()
    
    pass

# Muestra datos en 3D
def Plot3D(v, labels, nombre_ejes):
    # Figuar
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # Colores
    colors = ['red','green','blue','purple','yellow','black','orange','gray','aqua','brown']
        
    for n in range(10):
        data = v[np.nonzero(labels == n)]
        # Separar caracteristicas por eje
        x = [i[0] for i in data]
        y = [i[1] for i in data]
        z = [i[2] for i in data]
        # Dibujar puntos
        ax.scatter(x, y, z, c=colors[n])

    ax.set_xlabel(nombre_ejes[0])
    ax.set_ylabel(nombre_ejes[1])
    ax.set_zlabel(nombre_ejes[2])
    plt.show()
    
    pass

# Muestra datos en 3D
def Plot3D_2(x, y, z, nombre_ejes):
    # Figuar
    fig = plt.figure()
    ax = Axes3D(fig)
    

    # Dibujar puntos
    ax.scatter(x, y, z, c='green')

    ax.set_xlabel(nombre_ejes[0])
    ax.set_ylabel(nombre_ejes[1])
    ax.set_zlabel(nombre_ejes[2])
    plt.show()
    
    pass

# Reducimos la dimensionalidad del problema para simplificarlo.
# Se buscan aquellas caracteristicas que no tengan relevancia.
# Para ello se comprueba si existen caracteristicas que tengan media casi
# similar
def FiltrarDatos(x, labels):
    # Separar por digitos y obtener la media y varianza de cada caracteristica
    data = np.array([None] * 10)
    data_med = np.array([None] * 10)
    data_var = np.array([None] * 10)
    for n in range(10):
        data[n] = x[np.nonzero(labels == n)]
        data_med[n] = data[n].mean(axis=0)
        data_var[n] = data[n].var(axis=0)
    
    # Obtener la varianza de las medias
    data_med_var = data_med.mean(axis=0)
    
    # Obtener el minimo de las varianzas
    mins = np.array([None] * 10)
    
    for n in range(10):
        mins[n] = data_var[n].min()
    
    # Descartar segun umbral
    umbral = 10
    caracteristicas_a_mantener = data_med_var > umbral
    
    cols = []
    cols_quitar = []
    for i in range(len(caracteristicas_a_mantener)):
        if(caracteristicas_a_mantener[i]):
            cols.append(i)
        else:
            cols_quitar.append(i)
    
    return cols, cols_quitar

# knn con leave one out
def KnnLOU(x, y):
    
    grupos = [None] * 10
    
    # Separamos por digitos y nos quedamos con el 10% de los indices
    for n in range(10):
        grupos[n] = np.nonzero(y == n)[0]
        size10 = int(len(grupos[n])/10)
        grupos[n] = grupos[n][0:size10]
    
    aciertos = 0
    total    = 0
    
    # Recorrer cada grupo
    for n in range(10):
        # Recorrer todos los datos del grupo
        for i in grupos[n]:
            # Obtener las diferencias
            difs = [0] * len(x)
            
            for j in range(len(x)):
                # No usarse a si mismo
                if(i == j):
                    difs[j] = 10000000 # Valor infinito
                else:
                    difs[j] = np.sum(np.abs(np.subtract(x[i], x[j])))
    
            # Ordenar de menor a mayor por indices
            difs = np.argsort(difs)
            
            # Quedarse con el primero
            label = y[difs[0]]
    
            # Comprobar
            if(label == n):
                aciertos = aciertos+1
                
            total = total+1
            
    return round((float(aciertos)/float(total))*100.0, 2)

def PlotHistograma(x, y):
    # Separar por digitos y obtener la media de cada caracteristica
    data = np.array([None] * 10)
    data_med = np.array([None] * 10)
    for n in range(10):
        data[n] = x[np.nonzero(y == n)]
        data_med[n] = data[n].mean(axis=0)
        
    x = np.arange(16)
    
    colors=[''] * 16
    
    for i in range(8):
        colors[i]   = 'blue'
        colors[i+8] = 'red'
    
    # plot with various axes scales
    fig, axs = plt.subplots(2, 5, constrained_layout=True)

    for n in range(10):
        axs[int(n/5)][n%5].set_title('Digito '+str(n))
        axs[int(n/5)][n%5].set_ylim(0, 120)
        axs[int(n/5)][n%5].bar(x, height = data_med[n], color=colors)
        axs[int(n/5)][n%5].set_xlabel('Características')
        axs[int(n/5)][n%5].set_ylabel('Intensidad')
        
        axs[int(n/5)][n%5].set_xticklabels(['1', '9', '16'])
        axs[int(n/5)][n%5].set_xticks(range(0, 17, 8))
    
    plt.show()
    
    pass

def Ejercicio1():
    print("#######################")
    print("# DIGITOS MANUSCRITOS #")
    print("#######################")
    print()
    
    # Cargamos los datos de training de digitos
    print("Cargando conjunto de training optdigits.tra...")
    x_train, y_train = CargarDatos("datos/optdigits.tra", "int", ',')
    
    # Mostramos 10x10 digitos
    print("Mostramos algunos de los elementos")
    MostrarNumeros(x_train, y_train)
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Evaluamos KNN
    print("Tasa de aciertos Knn para 64D... ")
    aciertos = KnnLOU(x_train, y_train)
    print(str(aciertos)+"%")
    
    # Los convertimos a 3D y los visualizamos
    print("Convirtiendo a 3D...")
    x3D_train = DataA3D(x_train)
    
    print("Plot 3D")
    Plot3D(x3D_train, y_train, ["Simetría vertical", "Intensidad superior", "Intensidad inferior"])
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Evaluamos la tasa de aciertos del conjunto
    print("Tasa de aciertos Knn para 3D...")
    aciertos = KnnLOU(x3D_train, y_train)
    print(str(aciertos)+"%")
    
    # Convertimos los datos a 16D
    x16D_train = DataA16D(x_train)
    
    # Evaluamos la tasa de aciertos del conjunto
    print("Tasa de aciertos Knn para 16D...")
    aciertos = KnnLOU(x16D_train, y_train)
    print(str(aciertos)+"%")
    
    PlotHistograma(x16D_train, y_train)
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Filtrar no importantes
    print("Filtrando caractersíticas no importantes...")
    cols, cols_quitar = FiltrarDatos(x16D_train, y_train)
    print("Dimensión reducida de 16 a "+str(len(cols)))
    print("Se han eliminado las caracteristicas "+str(np.array(cols_quitar)+1))
    
    # Conjunto de dimension reducida
    xf_train = [i[cols] for i in x16D_train]
    
    # Evaluamos la tasa de aciertos del conjunto
    print("Tasa de aciertos Knn para "+str(len(cols))+"D")
    aciertos = KnnLOU(xf_train, y_train)
    print(str(aciertos)+"%")
    
    # Usamos Multinomial Logistic Regression de Sk Learn con Newton
    # 
    print("Entrenando clasificador...")
    softReg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=200)
    softReg.fit(xf_train, y_train)
    
    # Cargamos los datos de test de digitos
    print("Cargando conjunto de test optdigits.tes...")
    x_test, y_test = CargarDatos("datos/optdigits.tes", 'int', ',')
    
    # Transformamos y reducimoslos datos de test
    print("Transformando y reduciendo datos de test...")
    x16D_test = DataA16D(x_test)
    xf_test = [i[cols] for i in x16D_test]
    
    # Obtenemos las predicciones de clase
    print("Calculando predicciones...")
    predicciones = softReg.predict(xf_test)
    
    # Por cada prediccion calculamos el error
    print("Calculando error (tasa de aciertos)...")
    aciertos = 0.0
    for i in range(len(y_test)):
        if predicciones[i] == y_test[i]:
            aciertos = aciertos + 1.0
    
    e_out = 1.0-aciertos/len(y_test)
    
    print("e_out = ", e_out)
    print("Tasa de aciertos: "+str(round((aciertos/len(y_test))*100.0, 2))+"%")
    
    predicciones = softReg.predict(xf_train)
    aciertos = 0.0
    for i in range(len(y_train)):
        if predicciones[i] == y_train[i]:
            aciertos = aciertos + 1.0
    
    e_in = 1.0-aciertos/len(y_train)
    
    print("e_in = ", e_in)
    print("Tasa de aciertos: "+str(round((aciertos/len(y_train))*100.0, 2))+"%")
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    pass

# Get Minibatches
def getMinibatches(x, y, M, seed):
    
    # Copiar datos
    X = np.copy(x)
    Y = np.copy(y)

    # Barajar
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)

    # Dividir en arrays de tamano M
    X = np.array_split(X, M)
    Y = np.array_split(Y, M)
    
    return X, Y

# Gradiente Descendente Estocastico
#
#   x: conjunto de datos x=(xi,yi)
#   M: numero de elementos a usar: M << N
#      se recomienda 32<=M<=128
#   max_iter: cuantas iteraciones hasta terminar
#
#   return: w y ein
def sgd(x, y, eta, M, max_iter):
    mini_size = math.ceil(len(x)/M) # Tamano de los minibatches
    iterations = 0 # Numero de iteraciones, 0 por ahora
    w  = np.array([0.0] * len(x[0]))
    ww = np.array([0.0] * len(x[0]))

    # Mientras el peso actual es mayor que el minimo deseado, aplicar el algoritmo
    while iterations < max_iter:
        
        # Minibatch actual
        mini = iterations % mini_size
        
        # Si mini es 0, nuevo grupo
        if(mini == 0):
            X, Y = getMinibatches(x, y, mini_size, iterations)
        
        # Para guardar la derivada de w
        dw = 0
        
        # Calculamos valor nuevo para todos los puntos
        for j in range(len(w)):
            for i in range(len(X[mini])):
                dw += X[mini][i][j]*(np.matmul(w.T, X[mini][i])-Y[mini][i])
            
            dw *= (2.0/M)
            ww[j] -= eta*dw
            
        # Actualizamos
        w = np.copy(ww)
        
        # Aumentamos el numero de iteraciones
        iterations+=1
        
    # Devolvemos
    return w, iterations

# Funcion para calcular el error
def Err(y, y_predicciones):
    
    # Tamano de los datos
    n = len(y)
    
    # Para guardar el error
    e = 0
    
    # Aplicamos el sumatorio
    for i in range(n):
        e += (y[i]-y_predicciones[i])**2.0
    
    # La division entre n
    e /= n
    
    return e

def Ejercicio2():
    print("###########")
    print("# AIRFOIL #")
    print("###########")
    print()
    
    # Los nombres de cada caracteristica
    nombres_caracteristicas = [
    'Frecuencia (Hz)', \
    'Ángulo de ataque (grados)', \
    'Longitud de cuerda (m)', \
    'Velocidad de corriente libre (m/s)', \
    'Espesor de desplazamiento lateral de succión (m)', \
    'Presión del sonido (db)']
    
    # Cargamos los datos de airfoil
    print("Cargando datos airfoil_self_noise.dat...")
    x, y = CargarDatos("datos/airfoil_self_noise.dat", "float", '\t')
    
    # Hacemos random y separamos en 80% training y 20% test
    # Usamos la misma semilla para evitar que las entradas tengan
    # salidas que no les corresponden
    print("Barajamos los datos...")
    np.random.seed(1)
    np.random.shuffle(x)
    np.random.seed(1)
    np.random.shuffle(y)
    
    print("Separando en training (80%) y test (20%)...")
    p80  = int(float(len(x))*0.8)
    p100 = len(x)
    
    x_train = x[0:p80]
    y_train = y[0:p80]
    
    x_test = x[p80:p100]
    y_test = y[p80:p100]
    
    # Separamos por caracteristica
    print("Separamos por caracteristica y graficamos cada pareja con la salida:")
    x_train_caracteristica = [None] * len(x_train[0])
    for j in range(len(x_train[0])):
        x_train_caracteristica[j] = [i[j] for i in x_train]
    
    # Plots 3D de cada combinacion y la salida
    for i in range(5):
        for j in range(i, 5):
            if i != j:
                Plot3D_2(x_train_caracteristica[i], x_train_caracteristica[j], y_train, [nombres_caracteristicas[i], nombres_caracteristicas[j], nombres_caracteristicas[5]])
                #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Graficamos los datos de training
    print("Graficamos la salida")
    Plot2Ds([y_train], "Elementos", nombres_caracteristicas[5], sort='No')
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # No se ven tendencias, los mostramos ordenados de menor a mayor
    print("Graficamos la salida ordenada de menor a mayor")
    Plot2Ds([y_train], "Elementos", nombres_caracteristicas[5])
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Normalizar datos entre 0 y 1
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    y_train = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))
    
    # Antes de aplicar SGD, anadimos 1 a la primera columna
    # de cada tupla de datos por el termino independiente
    print("Ejecutamos SGD con eta=0.01, M=4, max_iter=10.000: ")
    x_train = [[1.0, i[0], i[1], i[2], i[3], i[4]] for i in x_train]
    
    w, it = sgd(x_train, y_train, 0.01, 4, 10000)
    
    y_predicciones = [0.0] * len(y_train)
    for i in range(len(y_train)):
        y_predicciones[i] = np.matmul(w.T, x_train[i])
        
    ein = Err(y_train, y_predicciones)
    
    # Datos de la ejecucion
    print("w = ", w)
    print("Nº iteraciones: ", it)
    print()
    
    # Vemos el error de entrada
    print("Ein: ", ein)
    
    # Repetimos el proceso de normalizar y anadir 1 pero con los datos de salida
    x_test = min_max_scaler.fit_transform(x_test)
    y_test = (y_test-np.min(y_test))/(np.max(y_test)-np.min(y_test))
    
    x_test = [[1.0, i[0], i[1], i[2], i[3], i[4]] for i in x_test]
    
    # Vemos el error de salida
    
    y_predicciones_test = [0.0] * len(y_test)
    for i in range(len(y_test)):
        y_predicciones_test[i] = np.matmul(w.T, x_test[i])

    eout = Err(y_test, y_predicciones_test)
    print("Eout: ", eout)
    
    # Graficamos la salida y el ajuste
    print("Graficamos el ajuste de entrada y salida ")
    Plot2Ds([y_train, y_predicciones], "Elementos", nombres_caracteristicas[5])
    
    Plot2Ds([y_test, y_predicciones_test], "Elementos", "db")
    
    #input("\n--- Pulsar tecla para continuar ---\n")
    
    # Usamos Random Forest como comparacion, calculando el error de entrada y salida
    print("Ejecutamos Random Forest Regressor con 100 árboles")
    model = RandomForestRegressor(n_estimators=100, max_features=6)
    model.fit(x_train, y_train)
    
    y_predicciones = model.predict(x_train)
    eout = Err(y_train, y_predicciones)
    print("Eout: ", eout)
    
    y_predicciones_test = model.predict(x_test)
    eout = Err(y_test, y_predicciones_test)
    print("Eout: ", eout)
    
    print("Graficamos el ajuste de entrada y salida ")
    Plot2Ds([y_train, y_predicciones], "Elementos", nombres_caracteristicas[5])
    Plot2Ds([y_test, y_predicciones_test], "Elementos", nombres_caracteristicas[5])
    
    pass

# Ejecutamos los ejercicios
Ejercicio1()
Ejercicio2()