import pickle
import os
from rtree import index


def build_rtree_index(data):
    p = index.Property()
    p.dimension = len(data[0])  # Dimensionality de los vectores característicos
    idx = index.Index(properties=p)
    
    for i, vector in enumerate(data):
        idx.insert(i, tuple(vector))
    
    return idx

def save_index_to_file(idx, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, 'wb') as file:
        print(f"Guardado en: {file_path}")
        pickle.dump(idx, file)


data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Vectores característicos de las imágenes

# Construir el índice R-tree
idx = build_rtree_index(data)

print("GUARDAR INDICES")
print(idx)
save_index_to_file(idx, 'indices.pkl')


