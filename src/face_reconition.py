import face_recognition
import os


directorio_coleccion_dev = "dataset/images_dev/"
directorio_coleccion_local = "dataset/images_local1/"

def characterist_image(imagen_path):

    imagen = face_recognition.load_image_file(imagen_path)
    caracteristicas = face_recognition.face_encodings(imagen)
    
    
    if len(caracteristicas) > 0: # Es rostro
        return caracteristicas[0]
    
    # No hay rostros
    return None

def characterist_all_images(directorio):
    caracteristicas_coleccion = []
    
    for imagen_nombre in os.listdir(directorio):
        imagen_path = os.path.join(directorio, imagen_nombre)
                
        caracteristicas = characterist_image(imagen_path)
        if caracteristicas is not None:
            caracteristicas_coleccion.append(caracteristicas)
    
    return caracteristicas_coleccion

def characterist_all_images_dir(directorio_raiz):
    caracteristicas_coleccion = []
    
    for nombre_persona in os.listdir(directorio_raiz):
        directorio_persona = os.path.join(directorio_raiz, nombre_persona)
        
        if os.path.isdir(directorio_persona):
            for imagen_nombre in os.listdir(directorio_persona):
                imagen_path = os.path.join(directorio_persona, imagen_nombre)                
                caracteristicas = characterist_image(imagen_path)
                if caracteristicas is not None:
                    caracteristicas_coleccion.append(caracteristicas)
    
    return caracteristicas_coleccion

#caracteristicas_coleccion_dev = characterist_all_images(directorio_coleccion_dev)
#directorio_coleccion_local = characterist_all_images_dir(directorio_coleccion_local)


#print("Se extrajeron características de", len(caracteristicas_coleccion_dev), "rostros en la colección.")
#print("Se extrajeron características de", len(directorio_coleccion_local), "rostros en la colección.")
