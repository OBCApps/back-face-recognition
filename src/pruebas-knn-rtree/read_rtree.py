import pickle
import os
from rtree import index


def load_index_from_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, 'rb') as file:
        idx = pickle.load(file)
    
    return idx


def knn_rtree(query, idx, data, k):
    result = list(idx.nearest(tuple(query), k))
    return [(data[i], i) for i in result]


print("CARGAR INDICES")
loaded_idx = load_index_from_file('indices.pkl')

query = [2, 3, 4]  # Imagen de consulta
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#Realizar una búsqueda KNN utilizando el índice cargado
knn_results = knn_rtree(query, loaded_idx, data, k=2)

#Imprimir los resultados
for result in knn_results:
    print("Imagen:", result[0])
    print("Índice:", result[1])
    print()