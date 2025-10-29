

import cv2
from deepface import DeepFace

# --- Inicialización ---

# Inicializa la captura de video desde la cámara web predeterminada (dispositivo 0)
cap = cv2.VideoCapture(0)

# Verificador para asegurar que la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# --- Variables para Optimización ---

# Contador de fotogramas para controlar la frecuencia del análisis
frame_counter = 0
# Almacena los resultados del último análisis exitoso para dibujarlos en fotogramas intermedios
previous_results = []

# --- Bucle Principal ---

while True:
    # Lee un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma. Saliendo...")
        break

    # Incrementa el contador de fotogramas en cada iteración
    frame_counter += 1

    # --- Lógica de Optimización ---
    # Se ejecuta el análisis de DeepFace solo una vez cada 10 fotogramas
    if frame_counter % 10 == 0:
        try:
            # Realiza el análisis facial para obtener edad y género.
            # Se especifica el backend 'mtcnn' para la detección de caras.
            # El parámetro enforce_detection=False evita que el programa se detenga si no hay caras.
            results = DeepFace.analyze(
                frame, 
                actions=['age', 'gender'], 
                enforce_detection=True, 
                detector_backend='mtcnn'
            )
            
            # Si el análisis es exitoso, actualizamos nuestros resultados previos
            previous_results = results
            
        except ValueError:
            # Este bloque se ejecuta si DeepFace.analyze no encuentra ninguna cara.
            # No hacemos nada y simplemente continuamos, el programa seguirá mostrando
            # los resultados del último análisis válido.
            # Si queremos que los cuadros desaparezcan cuando no se detecta cara,
            # podríamos limpiar la lista aquí: previous_results = []
            pass

    # --- Visualización ---
    # Dibuja los resultados en CADA fotograma, usando los datos almacenados en 'previous_results'.
    # Esto asegura que la visualización sea constante y fluida, incluso cuando el análisis no se está ejecutando.
    if previous_results:
        for face_data in previous_results:
            # Extrae la región facial (el rectángulo que enmarca la cara)
            facial_area = face_data['region']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Extrae el género y la edad
            gender_data = face_data['gender']
            gender = max(gender_data, key=gender_data.get)
            age = face_data['age']
            
            # Traduce el género para una mejor visualización
            gender_es = "Mujer" if gender == "Woman" else "Hombre"

            # Dibuja el rectángulo alrededor de la cara
            # El color es (0, 255, 0) que es verde, y el grosor es 2 píxeles.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepara el texto para mostrar (Género y Edad)
            text = f"{gender_es}, {age} anios"

            # Dibuja un fondo para el texto para mejorar la legibilidad
            # Obtenemos el tamaño del texto para dibujar un rectángulo relleno
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y - 5), (0, 255, 0), -1)


            # Escribe el texto (género y edad) justo encima del rectángulo
            # La posición es (x, y - 10) para que aparezca sobre el cuadro.
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Muestra el fotograma resultante en una ventana llamada 'Analisis Facial en Tiempo Real'
    cv2.imshow('Analisis Facial en Tiempo Real', frame)

    # --- Condición de Salida ---
    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza ---
# Libera el objeto de captura de video
cap.release()
# Cierra todas las ventanas de OpenCV
cv2.destroyAllWindows()
