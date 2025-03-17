import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Definir el dispositivo para YOLO
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n-pose.pt').to(device)

# Iniciar RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

# Puntos clave de interés (según COCO)
KEYPOINTS = {
    'right_ankle': 15,
    'left_ankle': 16,
    'right_hip': 11,
    'left_hip': 12
}

# Inicializar Kalman Filters para cada punto clave en 3D
def create_kalman_filter():
    kf = KalmanFilter(dim_x=6, dim_z=3)  # Estado (x, y, z, vx, vy, vz), medición (x, y, z)
    kf.F = np.array([[1, 0, 0, 1, 0, 0],  # Modelo de transición de estado
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    kf.H = np.array([[1, 0, 0, 0, 0, 0],  # Relación entre estado y observaciones
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

    kf.P *= 1000  # Covarianza inicial grande
    kf.R = np.eye(3) * 0.5  # Ruido de medición
    kf.Q = np.eye(6) * 0.01  # Ruido del proceso
    return kf

kalman_filters = {k: create_kalman_filter() for k in KEYPOINTS}

# Variables para cálculos
prev_positions = {k: None for k in KEYPOINTS}
prev_time = None

while True:
    # Capturar datos de RealSense
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Realizar detección con YOLOv8 Pose
    results = model(color_image, conf=0.5)
    
    if len(results[0].keypoints) == 0:
        continue  # Si no hay detecciones, pasamos al siguiente frame

    # Obtener tiempo actual para cálculos de velocidad/aceleración
    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    for detection in results[0].keypoints:
        keypoints = detection.xy.cpu().numpy().astype(int)  # Extraer coordenadas 2D
        
        for key, idx in KEYPOINTS.items():
            x, y = keypoints[idx]

            # Obtener profundidad de RealSense
            depth = depth_frame.get_distance(x, y) * 100  # Convertir a cm
            
            if depth == 0:  
                continue  # Si no hay datos válidos de profundidad, saltamos

            current_position = np.array([x, y, depth])

            if prev_positions[key] is not None and prev_time is not None:
                dt = current_time - prev_time
                velocity = (current_position - prev_positions[key]) / dt
                acceleration = velocity / dt  # Aceleración = ΔV / Δt

                # Aplicar filtro de Kalman
                kalman = kalman_filters[key]
                kalman.predict()
                kalman.update(current_position)
                smoothed_position = kalman.x[:3]  # Extraer posición suavizada

                # Dibujar resultados
                cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(color_image, f"{key}: {smoothed_position[2]:.1f} cm",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Actualizar valores previos
            prev_positions[key] = current_position

    prev_time = current_time

    # Mostrar imágenes
    cv2.imshow('Color Image', color_image)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    cv2.imshow('Depth Image', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()
