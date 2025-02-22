from ultralytics import YOLO

# Cargar el modelo entrenado (cambia 'best.pt' por el nombre de tu modelo)
model = YOLO("yolov8n.pt")  

# Exportar a ONNX
model.export(format="onnx")  