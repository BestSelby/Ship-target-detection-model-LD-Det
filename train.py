import sys
sys.path.append("F:/ultralytics-main")
from ultralytics import YOLO

if __name__ == '__main__':

 model = YOLO('ultralytics/cfg/models/v8/yolov8n-LD.yaml')  # load a pretrained moSdel (recommended for training)
 savename = 'train_yolo8n-Loss'
 results = model.train(data='datasets/Ship/ship.yaml', epochs=100,batch=16,imgsz=640,name=savename)

#yolo detect train model=yolov8n.yaml data=datasets/Ship/ship.yaml;