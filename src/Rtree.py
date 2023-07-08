import os
from rtree import index
from queue import PriorityQueue
import face_recognition
import numpy as np

from face_reconition import characterist_all_images_dir ,  characterist_image
directory_dataset = "dataset/images_local1/"
directory_imput = "input/"



def crear_indice_rtree(directorio_raiz):
    caracteristicas_coleccion = characterist_all_images_dir(directorio_raiz)

    idx = index.Index()
    for i, caracteristicas in enumerate(caracteristicas_coleccion):
        bbox = (caracteristicas[0], caracteristicas[1], caracteristicas[0], caracteristicas[1])
        idx.insert(i, bbox)

    return idx




def search_rtree(rtree_index, consulta_imagen_path, k, data):
    consulta_caracteristicas = characterist_image(consulta_imagen_path)
    caracteristicas_coleccion = characterist_all_images_dir(data)
    # Definir un rango de búsqueda adecuado
    rango_busqueda = 0.5  # Puedes ajustar este valor según tus necesidades
    bbox = (
        consulta_caracteristicas[0] - rango_busqueda,
        consulta_caracteristicas[1] - rango_busqueda,
        consulta_caracteristicas[0] + rango_busqueda,
        consulta_caracteristicas[1] + rango_busqueda
    )

    resultados = rtree_index.intersection(bbox, objects=False)
    resultados = list(resultados)  # Convertir a lista de índices

    distancias = face_recognition.face_distance([consulta_caracteristicas], [caracteristicas_coleccion[i] for i in resultados])
    resultados_ordenados = sorted(zip(resultados, distancias), key=lambda x: x[1])  # Ordenar por distancia

    print(f"Resultados más cercanos ({k}):")
    for resultado, distancia in resultados_ordenados[:k]:
        print(resultado, "Distancia:", distancia)

    print("Fin")






rtree_index = crear_indice_rtree(directory_dataset)
search_rtree(rtree_index, directory_imput + "uno1.JPG", 5 , directory_dataset)
print(rtree_index)