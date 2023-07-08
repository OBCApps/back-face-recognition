import json

def load_indices(file_path):
    # Cargar el archivo de texto
    with open(file_path, 'r') as file:
        indices_json = file.read()

    # Deserializar los índices desde JSON
    indices_list = json.loads(indices_json)

    # Convertir los índices a un conjunto
    indices_set = set(indices_list)

    return indices_set


# Especifica el archivo donde se guardaron los índices
file_path = "indices.txt"

# Cargar los índices desde el archivo de texto
loaded_indices = load_indices(file_path)

# Realizar operaciones con los índices cargados
for indice in loaded_indices:
    print(indice)
