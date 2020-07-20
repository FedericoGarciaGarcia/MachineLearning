def sgdRL(x, y, eta, M):
    mini_size = math.ceil(len(x)/M) # Tamano de los minibatches
    iterations = 0 # Numero de iteraciones, 0 por ahora
    w    = np.array([0.0] * 3) # W actual
    ww   = np.array([0.0] * 3) # W que se va actualizando
    want = np.array([0.0] * 3) # W anterior

    # Mientras la diferencia de pesos sea mayor o igual a 0.01
    seguir = True
    while seguir:
        
        # Minibatch actual
        mini = iterations % mini_size
        
        # Si mini es 0, nuevo grupo
        if(mini == 0):
            # Generar nuevos minibatches
            X, Y = getMinibatches(x, y, mini_size, iterations)
            
        # Calculamos valor nuevo para todos los puntos
        for j in range(len(w)):
            # Para guardar la derivada de w
            dw = 0
            # Formula
            for i in range(len(X[mini])):
                dw += (Y[mini][i]*X[mini][i][j]) / \
                (1+math.exp((np.matmul(w.T, X[mini][i]))*Y[mini][i]))
            dw *= -(1.0/M)
            
            # Actualizar usando el eta
            ww[j] -= eta*dw
             
        # Aumentamos el numero de iteraciones
        iterations+=1
        
        # Comprobar si se ha dado una vuelta completa 
        # (se han leido todos los datos)
        if(iterations % mini_size == 0):
            # Actualizamos
            w = np.copy(ww)
            # Si la diferencia es menos del 0.01 con el w de la
            # epoca anterior, parar
            diff = np.abs(w-want)
            if(diff[0] < 0.01 and diff[1] < 0.01 and diff[2] < 0.01):
                seguir = False
            # Guardar copia en antiguo
            want = np.copy(w)
        
    # Devolvemos
    return w, iterations
