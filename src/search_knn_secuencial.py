
from face_reconition import characterist_image , characterist_all_images, characterist_all_images_dir
import numpy as np
import heapq

directory_dataset = "dataset/images_local1/"
directory_imput = "input/"


def knn_secuencial_pq(objeto_consulta, caracteristicas_coleccion, k):
    pq = []
    
    # Calcular la distancia entre el objeto de consulta y cada vector de características de la colección
    for i, caracteristicas in enumerate(caracteristicas_coleccion):
        distancia = np.linalg.norm(objeto_consulta - caracteristicas)
        heapq.heappush(pq, (distancia, i))
        
        # Mantener el tamaño de la cola de prioridad en K
        if len(pq) > k:
            heapq.heappop(pq)
    
    # Obtener los K vecinos más cercanos
    vecinos_cercanos = [i for _, i in pq]
    
    # Ordenar los vecinos por distancia ascendente
    vecinos_cercanos.sort()
    
    return vecinos_cercanos

def knn_secuencial_rango(objeto_consulta, caracteristicas_coleccion, radio):
    vecinos_rango = []
    
    # Calcular la distancia entre el objeto de consulta y cada vector de características de la colección
    for i, caracteristicas in enumerate(caracteristicas_coleccion):
        distancia = np.linalg.norm(objeto_consulta - caracteristicas)
        
        # Si la distancia está dentro del rango, agregar el vecino a la lista
        if distancia <= radio:
            vecinos_rango.append(i)
    
    return vecinos_rango


query = characterist_image(directory_imput + 'uno1.JPG')
caracteristicas_coleccion = characterist_all_images_dir(directory_dataset)

k = 7
radio = 0.1

vecinos_cercanos_pq = knn_secuencial_pq(query, caracteristicas_coleccion, k)
vecinos_cercanos_rango = knn_secuencial_rango(query, caracteristicas_coleccion, radio)

print("vecinos_cercanos_pq  " , vecinos_cercanos_pq )
print("vecinos_cercanos_rango  " , vecinos_cercanos_rango )