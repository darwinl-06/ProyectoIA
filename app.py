from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Cargar modelo, escalador, PCA y encoder
svm_model = joblib.load('Entrega3/svm_model.pkl')
scaler = joblib.load('Entrega3/scaler.pkl')
pca = joblib.load('Entrega3/pca.pkl')

# Cargar clases (etiquetas)
try:
    le = joblib.load('Entrega3/label_encoder.pkl')
    label_classes = le.classes_
except:
    with open('Entrega3/label_classes.pkl', 'rb') as f:
        label_classes = pickle.load(f)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Configurar un buffer para almacenar características de 5 frames
frame_buffer = []

def extract_features(landmarks):
    features = []
    for lm in landmarks:
        features.extend([float(lm.x), float(lm.y), float(lm.z)])
    return features

def generate_frames():
    global frame_buffer
    cap = cv2.VideoCapture(0)  # Accede a la cámara
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        # Convertir el frame a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extraer características del frame actual
            features = extract_features(landmarks)

            # Agregar al buffer
            frame_buffer.append(features)
            if len(frame_buffer) > 5:
                frame_buffer.pop(0)

            # Procesar si el buffer tiene 5 frames
            if len(frame_buffer) == 5:
                try:
                    input_features = np.array(np.concatenate(frame_buffer), dtype=np.float64)
                    scaled_features = scaler.transform([input_features])
                    reduced_features = pca.transform(scaled_features)
                    prediction = svm_model.predict(reduced_features)[0]

                    # Mapear predicción a etiqueta
                    if isinstance(prediction, str):
                        activity = prediction
                    else:
                        activity = label_classes[int(prediction)] if int(prediction) < len(label_classes) else "Desconocido"

                    # Mostrar actividad en el frame
                    cv2.putText(frame, f'Actividad: {activity}', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error en el procesamiento: {str(e)}")
                    continue

            # Dibujar landmarks en el frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Codificar el frame para el stream
        ret, encoded_frame = cv2.imencode('.jpg', frame)
        frame = encoded_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint del video en streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)