def ajusta_PLA(datos, label, max_iter, wini):
    # Booleano para indicar si ha habido al menos un cambio
    # en un pase completo de datos
    cambio=True
    
    # El w y otro para guardar los cambios nuevos
    w       = [i for i in wini]
    w_nuevo = [i for i in wini]
    
    # Mientras queden iteraciones o en un pase no se haya conseguido mejorar
    it=0
    while(it < max_iter and cambio):
        # Por ahora no hay cambio
        cambio=False
        # Por cada elemento
        for i in range(0, len(datos)):
            if(signo(np.matmul(w, np.array(datos[i]).T)) != label[i]):
                # Por cada caracteristica
                for j in range(len(w)):
                    w_nuevo[j]=w[j]+datos[i][j]*label[i]
                    
                # Hubo cambio
                cambio = True
                
        # Actualizar los cambios
        w = [i for i in w_nuevo]
        
        # Siguiente iteracion
        it=it+1
            
    # Devolver los pesos e iteraciones
    return w, it
