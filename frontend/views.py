"""
Routes and views for the flask application.
"""
# -*- coding: utf-8 -*-
from datetime import datetime
from flask import render_template,request,Flask
from fruitProject import app

import requests
import cv2
@app.route('/')
@app.route('/Main',methods=['POST','GET'])
def Display():
    img_name=""
    if request.method=="POST":
        ori_img=request.files.get('ori_img')
        file_name=ori_img.filename
        pre="./fruitProject/static/"
        img_path=pre+f"{file_name}"
        if ori_img:
            ori_img.save(img_path)
            print("OK")
            resp=Analyze(img_path)
        
            return render_template(
                'Display.html',
                title="DisplayPage",
                image_name=resp
            )
    return render_template(
        'Display.html',
        title="DisplayPage",
        image_name=img_name
        )

from ultralytics import YOLO
def Analyze(ori_img_path):
    model=YOLO("best.pt")
    results=model(ori_img_path)
    pred_img_path=r"D:\fruitProject\fruitProject\fruitProject\static\res1.jpg"
    results[0].save(pred_img_path)
    rel_img_path="res1.jpg"
    return rel_img_path