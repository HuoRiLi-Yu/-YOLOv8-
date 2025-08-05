import os
import glob
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
classes=['banana','orange','apple']

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_xml_to_txt(xml_folder, txt_folder):
    xml_files = os.listdir(xml_folder)
    for xml_file in xml_files:
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            print(w,h)
            if (w==0 or h==0):
                print(xml_file)
                with Image.open(os.path.join(xml_folder,xml_file[:-4]+'.jpg')) as img:
                    width, height = img.size
                    w=int(width)
                    h=int(height)
            with open(os.path.join(txt_folder, xml_file[:-4] + '.txt'), 'w') as txt_file:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    
                    if cls not in classes or int(difficult) == 1:
                        continue
                    obj_id=classes.index(cls)
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    if xmax > w:
                        xmax = w
                    if ymax > h:
                        ymax = h
                    b = (xmin, xmax, ymin, ymax)
                    bb = convert((w, h), b)
                    txt_file.write(f"{obj_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

# 使用方法
#convert_xml_to_txt(r"E:\Yolo\mydata\train_data2", r"E:\Yolo\mydata\train_data2")

def move_files_to_folder(source_folder, target_folder, file_extension):
    files = glob.glob(f"{source_folder}/*.{file_extension}")
    for file in files:
        try:
            shutil.move(file, target_folder)
            print(f"File {file} has been moved successfully to {target_folder}")
        except OSError as e:
            print(f"Error: {file} : {e.strerror}")

# 使用方法
move_files_to_folder(r"E:\Yolo\mydata\train_data2",r"E:\Yolo\mydata\train_xml2", 'xml')