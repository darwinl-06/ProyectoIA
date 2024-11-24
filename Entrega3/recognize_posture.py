import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle

# Cargar modelo, escalador, PCA y encoder
svm_model = joblib.load('Entrega3/svm_model.pkl')
scaler = joblib.load('Entrega3/scaler.pkl')
pca = joblib.load('Entrega3/pca.pkl')
try:
    le = joblib.load('Entrega3/label_encoder.pkl')
    label_classes = le.classes_
except:
    with open('Entrega3/label_classes.pkl', 'rb') as f:
        label_classes = pickle.load(f)

# Imprimir información de debug
print(f"Dimensiones del PCA: {pca.n_components_}")
print(f"Número de características que espera el scaler: {scaler.n_features_in_}")
print(f"Número de características que espera el modelo SVM: {svm_model.n_features_in_}")
print("Label classes:", label_classes)
print("Tipo de label_classes:", type(label_classes))

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Configurar un buffer para almacenar características de 5 frames
buffer = []

def extract_features(landmarks):
    features = []
    # Extraer coordenadas x, y, z y convertirlas explícitamente a float
    for lm in landmarks:
        features.extend([float(lm.x), float(lm.y), float(lm.z)])
    return features

# Generar nombres de características
feature_names = [f"feature_{i}" for i in range(495)]

# Configurar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extraer características del frame actual
        features = extract_features(landmarks)
        
        # Debug print para verificar número de características por frame
        if len(buffer) == 0:
            print(f"Número de características por frame: {len(features)}")

        # Agregar al buffer
        buffer.append(features)
        if len(buffer) > 5:
            buffer.pop(0)

        # Procesar únicamente si el buffer tiene 5 frames
        if len(buffer) == 5:
            try:
                # Concatenar las características de los 5 frames
                input_features = np.array(np.concatenate(buffer), dtype=np.float64)
                
                # Verificar dimensiones
                print(f"Dimensiones de input_features: {input_features.shape}")
                
                # Crear DataFrame con nombres de características
                input_features_named = dict(zip(feature_names, input_features))
                
                # Primero escalar los datos
                scaled_features = scaler.transform([input_features])
                
                # Luego aplicar PCA para reducir a 10 dimensiones
                reduced_features = pca.transform(scaled_features)
                
                # Realizar la predicción
                prediction = svm_model.predict(reduced_features)[0]
                
                # Manejar la predicción
                if isinstance(prediction, str):
                    activity = prediction
                else:
                    try:
                        prediction_index = int(prediction)
                        activity = label_classes[prediction_index]
                    except (IndexError, TypeError, ValueError):
                        # Si hay algún error al convertir o acceder al índice
                        if prediction in label_classes:
                            activity = prediction
                        else:
                            activity = "Desconocido"
                            print(f"Predicción no válida: {prediction}")

                # Mostrar actividad en el frame
                cv2.putText(frame, f'Actividad: {activity}', (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error en el procesamiento: {str(e)}")
                continue

        # Dibujar landmarks en el frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Mostrar el frame procesado
    cv2.imshow('Clasificación en Tiempo Real', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()