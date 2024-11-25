# Proyecto de Anotación de Video

## Integrantes

- Darwin Lenis  
- Juan Felipe Madrid  
- Jacobo Ossa  
- Luis Pinillos  

---

## Descripción del Proyecto

Este proyecto consiste en una herramienta de software diseñada para analizar actividades humanas específicas (como caminar, girar, sentarse y ponerse de pie) utilizando un sistema de video en tiempo real. El proyecto aprovecha tecnologías avanzadas de visión por computadora y aprendizaje automático para realizar clasificaciones precisas y análisis posturales en vivo.

El sistema utiliza **MediaPipe** para detectar landmarks del cuerpo y un modelo de **Máquina de Soporte Vectorial (SVM)** para clasificar las actividades basándose en características extraídas de la postura de una persona.

---

## Tecnologías y Librerías Utilizadas

- **OpenCV**: Para la captura de video y procesamiento de imágenes.  
- **MediaPipe**: Para la detección de landmarks corporales en tiempo real.  
- **NumPy**: Para manejar datos numéricos y realizar operaciones matriciales.  
- **scikit-learn (joblib)**: Para el preprocesamiento de datos, reducción de dimensionalidad (PCA) y clasificación con SVM.  
- **pickle**: Para cargar clases etiquetadas previamente entrenadas.  

---

## Instalación de Dependencias

### Requisitos Previos
- Python 3.7 o superior.
- Tener instalado un entorno virtual o gestor de dependencias como `pip`.

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/darwinl-06/ProyectoIA.git
   cd ProyectoIA
   ```

2. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución del Proyecto

1. Asegúrate de que tu cámara esté conectada.
2. Ejecuta el script principal:
   ```bash
   python app.py
   ```
3. La ventana mostrará el video en tiempo real con:
   - Los landmarks corporales detectados.
   - La actividad humana clasificada en la parte superior izquierda del video.

## Explicación del proyecto

https://www.youtube.com/watch?v=ILYi6atOou0
  
