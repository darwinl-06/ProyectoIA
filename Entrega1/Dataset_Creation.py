import cv2
import mediapipe as mp
import csv
import numpy as np

# Inicializamos MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Etiqueta de la acción (cambiar según el video)
action_label = "girando"

# Ruta del video
video_path = './Desktop/ProyectoIA/Entrega1/Videos/Girando.mp4'
cap = cv2.VideoCapture(video_path)

# Abrimos un csv para guardar los datos de las coordenadas de las articulaciones
with open('./Desktop/ProyectoIA/Entrega1/Dataset_Info/dataset_pose_girando.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escribimos el encabezado del csv
    header = ['label']
    for i in range(33):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    # Procesamos cada fotograma del video
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convertimos el fotograma de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el fotograma con mediapipe para detectar las articulaciones
        results = pose.process(rgb_frame)

        # Si se detectan las articulaciones
        if results.pose_landmarks:
            # Extraemos las coordenadas x, y, z de cada articulación
            landmarks = results.pose_landmarks.landmark
            # Etiqueta del movimiento
            row = [action_label]
            
            # Agregamos las coordenadas de los 33 puntos clave obtenidos de las articulaciones
            for landmark in landmarks:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
            
            # Escribimos la fila en el csv con cada una de las coordenadas obtenidas
            writer.writerow(row)

# Liberamos recursos usdos
cap.release()
cv2.destroyAllWindows()
