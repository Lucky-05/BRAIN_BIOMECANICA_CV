import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HRNET import HRNET
from HRNET.utils import ModelType, filter_person_detections
from HRNET.yolov6.YOLOv6 import YOLOv6 as PersonDetector
