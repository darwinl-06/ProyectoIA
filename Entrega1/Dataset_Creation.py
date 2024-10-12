import cv2
import mediapipe as mp
import csv
import numpy as np

# Inicializamos MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Etiqueta de la acción (puedes cambiarla dependiendo del video que proceses)
action_label = "Sentarse"

# Ruta del video (puedes cambiarla según el video que proceses)
video_path = './Desktop/ProyectoIA/Entrega1/Videos/Sentado_User4.mp4'
cap = cv2.VideoCapture(video_path)

# Abrimos un archivo CSV para guardar los datos del dataset
with open('./Desktop/ProyectoIA/Entrega1/Dataset_Info/dataset_sentarse_user4.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escribimos el encabezado del CSV
    header = ['label']
    for frame_idx in range(5):  # 5 frames agrupados
        for i in range(33):  # 33 puntos mapeados
            header += [f'x{frame_idx}_{i}', f'y{frame_idx}_{i}', f'z{frame_idx}_{i}']
    writer.writerow(header)

    # Variables para agrupar los frames
    group_of_frames = []
    frame_counter = 0

    # Procesamos cada frame del video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convertimos el frame de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el frame con MediaPipe para detectar las articulaciones
        results = pose.process(rgb_frame)

        # Si se detectan las articulaciones
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = []

            # Extraemos las coordenadas x, y, z de cada articulación
            for landmark in landmarks:
                frame_data.append(landmark.x)
                frame_data.append(landmark.y)
                frame_data.append(landmark.z)

            # Añadimos las coordenadas del frame actual al grupo de frames
            group_of_frames.append(frame_data)
            frame_counter += 1

            # Cuando se tienen 5 frames, concatenamos las coordenadas en un solo vector
            if frame_counter == 5:
                combined_vector = [action_label]  # Añadimos la etiqueta al inicio
                for frame in group_of_frames:
                    combined_vector.extend(frame)  # Concatenamos las coordenadas de los 5 frames

                # Escribimos el vector combinado en el CSV
                writer.writerow(combined_vector)

                # Reiniciamos el contador y el buffer de frames
                group_of_frames = []
                frame_counter = 0

# Liberamos los recursos utilizados
cap.release()
cv2.destroyAllWindows()
