from ultralytics import YOLO
import cv2
model=YOLO(r"E:\Yolo\runs\detect\train4\weights\best.pt")
results=model(r"E:\Yolo\mydata\my_test\6.jpg")
for result in results:
    result.show()
    result.save(filename=".\res\res_6.jpg")