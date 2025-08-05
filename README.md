# 基于YOLOv8的水果识别系统
完成一个水果识别系统，输入一张包含水果的图片，可以识别出对应的水果种类
# 本项目使用了开源大模型YOLOV8,相关环境依赖已经保存在以下地址中
https://pan.baidu.com/s/162bINYJUoshHap-iNrLOXQ?pwd=uvpj

提取码：uvpj

## 文件说明：
1. test中保存原始测试集
2. best.pt为训练出的水果识别模型
3. train1.py是训练文件，我们使用378张左右的图片进行了50次左右的训练
4. test1.py用于测试一些图片的识别效果
5. util1.py是一些有用的功能函数
6. predict.py是用于在测试集上进行模型测试的代码。
7. fronted是前端交互的代码文件
