import cv2
import os
import sys
from ultralytics import YOLO
from HRNET import HRNET
import numpy as np
from sort import Sort
from utils import ModelType

# Ruta del video local
video_path = "Movimiento 1 - Caminata.mp4"  # Reemplaza con el nombre de tu video
cap = cv2.VideoCapture(video_path)
tracker = Sort()

# Verificar las dimensiones del video de entrada
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar la salida de video con las mismas dimensiones que el video de entrada
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
out = cv2.VideoWriter('out_try.mp4', fourcc, 30, (width, height))  # Salida con la misma resolución

# Inicializar modelo de Pose Estimation
model_path = "HRNET_ONX/models/hrnet_coco_w48_384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.3)

# Inicializar modelo de detección de personas usando ultralytics YOLO
person_detector = YOLO("HRNET_ONX/HRNET/yolov8n.pt")  # Ruta al modelo YOLOv8n, puedes usar otro modelo si prefieres

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)

while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):  # Presiona 'q' para salir
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Detección de personas utilizando ultralytics
    results = person_detector(frame)

    # Filtrar detecciones para obtener solo las personas (clase 0)
    boxes, scores, class_ids = [], [], []
    
    for res in results:
        # Filtrar solo las detecciones de la clase 0 (personas) y confianza mayor a 0.65
        filtered_indices = np.where(res.boxes.cls.cpu().numpy() == 0)[0]  # Solo personas (clase 0)
        filtered_boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)  # Coordenadas de las cajas
        filtered_confidence = res.boxes.conf.cpu().numpy()[filtered_indices]  # Confianza de las detecciones

        # Filtrar cajas según confianza mayor a 0.65
        valid_boxes = filtered_boxes[filtered_confidence > 0.65]
        valid_confidence = filtered_confidence[filtered_confidence > 0.65]

        # Agregar las detecciones válidas a las listas
        boxes.append(valid_boxes)
        scores.append(valid_confidence)
        class_ids.append(np.zeros(len(valid_boxes)))  # Todos son de la clase "persona" (ID 0)

        # Usar SORT para el seguimiento de objetos
        tracks = tracker.update(valid_boxes)

        # Dibujar las cajas de seguimiento sobre el frame
        for track in tracks:
            xmin, ymin, xmax, ymax, track_id = map(int, track)
            # Dibujar caja con ID de seguimiento
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(track_id)}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Estimación de pose si se detecta alguna persona
    if len(boxes) > 0:
        # Convertir listas a formato adecuado para HRNet
        person_detections = [np.vstack(boxes), np.hstack(scores), np.hstack(class_ids)]
        total_heatmap, peaks = hrnet(frame, person_detections)

        # Dibujar la estimación de la pose sobre el frame
        frame = hrnet.draw_pose(frame)

    # Mostrar resultado en pantalla
    cv2.imshow("Model Output", frame)

    # Guardar frame en el archivo de salida
    out.write(frame)
    print("Frame guardado correctamente")  # Depuración para asegurarte de que el frame se está guardando

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
