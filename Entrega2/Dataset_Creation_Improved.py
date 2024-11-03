import cv2
import mediapipe as mp
import csv

# Inicializamos MediaPipe pose y face detection con parámetros ajustados
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5  # Reducimos el umbral para mayor sensibilidad
)

# Etiqueta de la acción
action_label = "Sentandose"

# Ruta del video
video_path = './Desktop/ProyectoIA/Entrega2/Videos/Sentado_User6.mp4'
cap = cv2.VideoCapture(video_path)

def is_facing_camera(landmarks):
    """
    Función mejorada para verificar si la persona está de frente
    utilizando múltiples puntos de referencia
    """
    # Obtenemos puntos clave adicionales
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    # Calculamos múltiples métricas
    shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    eye_width = abs(left_eye.x - right_eye.x)
    
    # Verificamos la visibilidad de puntos clave faciales
    face_visibility = (
        left_eye.visibility > 0.5 and
        right_eye.visibility > 0.5 and
        nose.visibility > 0.5
    )
    
    # Criterios más flexibles para determinar si está de frente
    is_facing = (
        shoulder_depth_diff < 0.15 and  # Más tolerante con la diferencia de profundidad
        shoulder_width > 0.08 and       # Más tolerante con el ancho de hombros
        eye_width > 0.02 and           # Verificación adicional de los ojos
        face_visibility                 # Verificación de visibilidad
    )
    
    return is_facing

# Abrimos un archivo CSV para guardar los datos del dataset
with open('./Desktop/ProyectoIA/Entrega2/Dataset_Info/dataset_sentado_user6.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escribimos el encabezado del CSV
    header = ['label']
    for frame_idx in range(5):
        for i in range(33):
            header += [f'x{frame_idx}_{i}', f'y{frame_idx}_{i}', f'z{frame_idx}_{i}']
        header.append(f'face_detected_{frame_idx}')
    writer.writerow(header)

    # Variables para agrupar los frames
    group_of_frames = []
    face_detection_status = []
    frame_counter = 0

    # Procesamos cada frame del video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertimos el frame a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el frame con MediaPipe pose y face detection
        pose_results = pose.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        # Verificamos si se detectó el rostro
        face_detected = False
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Lógica mejorada de detección de rostro
            face_detected = bool(face_results.detections) or is_facing_camera(landmarks)

            frame_data = []
            # Extraemos las coordenadas x, y, z de cada articulación
            for landmark in landmarks:
                frame_data.append(landmark.x)
                frame_data.append(landmark.y)
                frame_data.append(landmark.z)

            # Añadimos las coordenadas y el estado de detección de rostro
            group_of_frames.append(frame_data)
            face_detection_status.append(face_detected)
            frame_counter += 1

            # Cuando tenemos 5 frames, guardamos en el CSV
            if frame_counter == 5:
                combined_vector = [action_label]
                for i, frame in enumerate(group_of_frames):
                    combined_vector.extend(frame)
                    combined_vector.append('VERDADERO' if face_detection_status[i] else 'FALSO')

                writer.writerow(combined_vector)

                # Reiniciamos el contador y el buffer de frames
                group_of_frames = []
                face_detection_status = []
                frame_counter = 0

            # Visualización del frame con información de detección
            frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Dibujamos un rectángulo y texto para mostrar el estado de detección
            cv2.rectangle(frame_rgb, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.putText(
                frame_rgb,
                f"Face detected: {face_detected}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if face_detected else (0, 0, 255),
                2
            )
            
            # Mostramos el frame
            cv2.imshow('Frame', frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Liberamos los recursos
cap.release()
cv2.destroyAllWindows()