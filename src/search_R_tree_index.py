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

def buscar_indice(idx, query):
    resultados = idx.intersection(query, objects=True)
    k_elementos_cercanos = []
    
    for resultado in resultados:
        name, index_image, caracteristicas = resultado.object
        index_str = str(index_image).zfill(4)  
        
        k_elementos_cercanos.append((name ,index_str ))
    return k_elementos_cercanos


def buscar_knn_rtree(idx, query, k):
    resultados = idx.nearest(query, num_results=k, objects=True)
    k_elementos_cercanos = []
    
    for resultado in resultados:
        name, index_image, caracteristicas = resultado.object
        index_str = str(index_image).zfill(4)  
        
        k_elementos_cercanos.append((name, index_str))
    
    return k_elementos_cercanos


def search_rtree_indexed_all(query, data, radio):
    idx = crear_indice(data)
    coordenadas = (query[0] - radio, query[1] - radio, query[0] + radio, query[1] + radio)
    resultados  = buscar_indice(idx, coordenadas)
    return return_images(resultados)

def search_rtree_indexed_knn(query, data, radio, k):
    idx = crear_indice(data)
    coordenadas = (query[0] - radio, query[1] - radio, query[0] + radio, query[1] + radio)
    resultados  = buscar_indice(idx, coordenadas)
    return return_images(resultados)
