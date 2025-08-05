from ultralytics import YOLO
import cv2
model = YOLO(r"E:\Yolo\runs\detect\train3\weights\best.pt")
results=model.train(data="mydata/mydata.yaml", epochs=30, imgsz=640,save=True)