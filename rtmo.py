import cv2
import numpy as np
from mmdeploy_runtime import PoseDetector

pose_model_path = 'rtmo-m_body7_onnx'
device_name = 'cpu'
img_path = 'image.jpg'

pose_detector = PoseDetector(pose_model_path, device_name, device_id=0)

image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

poses = pose_detector(image)