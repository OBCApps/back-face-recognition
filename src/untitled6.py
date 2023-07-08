#Reak
import numpy as np
import heapq
import base64
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
directory_dataset = os.path.join(current_dir, "dataset", "images_local1")
directory_input = os.path.join(current_dir, "input")

# print(f"SEarchSecuencial: Ruta actual = {current_dir}" )
# print(f"SEarchSecuencial: directory_dataset = {directory_dataset}" )
# print(f"SEarchSecuencial: directory_input = {directory_input}" )

def knn_secuencial_pq(query, data, k):
    pq = []
    
    for nombre_persona, caracteristicas_persona in data.items():
        for i, caracteristicas in enumerate(caracteristicas_persona, start=1):  # Índices comienzan desde 1
            distancia = np.linalg.norm(query - caracteristicas)
            indice_str = str(i).zfill(4)  # Agregar ceros a la izquierda
            heapq.heappush(pq, (distancia, nombre_persona, indice_str))
            if len(pq) > k:
                heapq.heappop(pq)
    
    vecinos_cercanos = [(nombre_persona, indice) for _, nombre_persona, indice in pq]     
    vecinos_cercanos.sort()
    return vecinos_cercanos



def knn_secuencial_rango(query, data, radio, k):
    vecinos_rango = []
    distancias = []    
    for nombre_persona, caracteristicas_persona in data.items():
        for i, caracteristicas in enumerate(caracteristicas_persona, start=1):  # Índices comienzan desde 1
            distancia = np.linalg.norm(query - caracteristicas)
            
            if distancia <= radio:
                distancias.append(distancia)
                indice_str = str(i).zfill(4)  # Agregar ceros a la izquierda
                vecinos_rango.append((nombre_persona, indice_str))
    
    vecinos_rango = [vecino for _, vecino in sorted(zip(distancias, vecinos_rango))]
    vecinos_rango = vecinos_rango[:k]
    
    return vecinos_rango


def return_image(direccion_imagen):
    with open(direccion_imagen, "rb") as image_file:
        imagen_bytes = image_file.read()
        imagen_base64 = base64.b64encode(imagen_bytes).decode("utf-8")
    return imagen_base64

def return_images(lista_imagenes):
    imagenes_base64 = []
    for nombre, indice in lista_imagenes:
        nombre_persona = nombre
        indice_str = str(indice)
        direccion_imagen = os.path.join("src", "dataset", "images_local1", nombre_persona , f"{nombre_persona}_{indice_str}.jpg")
        print("direccion_imagen: ",direccion_imagen )
        imagen_base64 = return_image(direccion_imagen)
        imagenes_base64.append(imagen_base64)
    return imagenes_base64




# query = characterist_image(directory_input + 'uno1.JPG')
# caracteristicas_coleccion = characterist_all_images_dir(directory_dataset)

# k = 4
# radio = 5

# vecinos_cercanos_pq = knn_secuencial_pq(query, caracteristicas_coleccion, k)
# vecinos_cercanos_rango = knn_secuencial_rango(query, caracteristicas_coleccion, radio , k)
# imagenes_base_64 = return_images(vecinos_cercanos_rango)
# print("vecinos_cercanos_pq: ", vecinos_cercanos_pq)
# print("vecinos_cercanos_rango: ", vecinos_cercanos_rango)
# print(len(imagenes_base_64))
