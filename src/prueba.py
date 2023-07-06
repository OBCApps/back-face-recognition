import pickle

# Objeto a serializar
datos = {'nombre': 'John', 'edad': 30}

# Guardar el objeto serializado en un archivo
with open('datos.pickle', 'wb') as archivo:
    pickle.dump(datos, archivo)

# Obtener el objeto serializado desde el archivo
with open('datos.pickle', 'rb') as archivo:
    datos_serializados = pickle.load(archivo)

print(datos_serializados) 
