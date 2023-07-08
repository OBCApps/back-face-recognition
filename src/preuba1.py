from rtree import index

# Crea un índice espacial R-tree
punto_index = index.Index()

# Agrega puntos al índice
punto_index.insert(1, (2.5, 4.5, 2.5, 4.5))  # ID: 1, coordenadas: (2.5, 4.5)
punto_index.insert(2, (1.0, 3.0, 1.0, 3.0))  # ID: 2, coordenadas: (1.0, 3.0)
punto_index.insert(3, (4.0, 5.0, 4.0, 5.0))  # ID: 3, coordenadas: (4.0, 5.0)
punto_index.insert(4, (7.0, 8.0, 7.0, 8.0))  # ID: 4, coordenadas: (7.0, 8.0)
punto_index.insert(5, (6.0, 9.0, 6.0, 9.0))  # ID: 5, coordenadas: (6.0, 9.0)

ventana = (1, 7, 7, 9)  # Ventana: (1, 3) a (7, 9)
resultados = punto_index.intersection(ventana)
xmin, ymin, xmax, ymax = punto_index.bounds

# Imprime los resultados de la consulta
for resultado in resultados:    
    print(resultado)
    
    
# ----------------- YA funcionaaa 
from rtree import index
import os
from src.search_knn_secuencial import return_images

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_dataset = os.path.join(current_dir, "dataset", "images_local1")
dir_input = os.path.join(current_dir, "input")


def crear_indice(dataset):
    idx = index.Index()
    
    for name, characteristic in dataset.items():
        for i, caracteristicas in enumerate(characteristic , start=1):
            xmin = ymin = xmax = ymax = i  # Establece las coordenadas de la ventana como el índice actual
            idx.insert(i, (xmin, ymin, xmax, ymax), obj=(name, i, caracteristicas))
    
    return idx

def buscar_indice(idx, query, k):
    resultados = idx.intersection(query, objects=True)
    k_elementos_cercanos = []
    
    for resultado in resultados:
        name, index_image, caracteristicas = resultado.object
        index_str = str(index_image).zfill(4)  # Añadir ceros a la izquierda si es necesario
        
        k_elementos_cercanos.append((name ,index_str ))
    
    k_elementos_cercanos = sorted(k_elementos_cercanos, key=lambda x: x[1])[:k]
    return k_elementos_cercanos



    

def search_rtree_indexed(query, data, radio, k):
    idx = crear_indice(data)
    coordenadas = (query[0] - radio, query[1] - radio, query[0] + radio, query[1] + radio)
    resultados  = buscar_indice(idx, coordenadas , k)
    return return_images(resultados)
