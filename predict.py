import sys
sys.path.append("F:/ultralytics-main/ultralytics/nn/AddModules")
# 即 ultralytics文件夹 所在绝对路径
from ultralytics import YOLO
model=YOLO('runs/detect/train_yolov8n/weights/best.pt')
model.predict('datasets\Ship\JPEGImages_test',save=True,imgsz=640,conf=0.5)