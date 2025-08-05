from ultralytics import YOLO
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET
classes=['banana','orange','apple']
def read_boxes(xml_file):
    if xml_file.endswith('.xml'):
        box_list=[]
        cls_list=[]
        tree = ET.parse(xml_file)
        root = tree.getroot()
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
           
            box_list.append([int(xmin),int(ymin),int(xmax),int(ymax)])
            cls_list.append(obj_id)
        return cls_list,box_list
 
def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou
 
def show_table(res_iou,res_pr,res_r,res_map):
    print(f"{'Class':^10}"+f"{'IOU':^10}"+f"{'Precision':^10}"+f"{'Recall':^10}"+f"{'MAP50':^10}")
    print(f"{'All':^10}"+f"{res_iou[0]:^10.2f}"+f"{res_pr[0]:^10.2f}"+f"{res_r[0]:^10.2f}"+f"{res_map[0]:^10.2f}")
    print(f"{'Banana':^10}"+f"{res_iou[1]:^10.2f}"+f"{res_pr[1]:^10.2f}"+f"{res_r[1]:^10.2f}"+f"{res_map[1]:^10.2f}")
    print(f"{'Orange':^10}"+f"{res_iou[2]:^10.2f}"+f"{res_pr[2]:^10.2f}"+f"{res_r[2]:^10.2f}"+f"{res_map[2]:^10.2f}")
    print(f"{'Apple':^10}"+f"{res_iou[3]:^10.2f}"+f"{res_pr[3]:^10.2f}"+f"{res_r[3]:^10.2f}"+f"{res_map[3]:^10.2f}")
def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

def ap_per_class(
    tp, conf, pred_cls, target_cls, eps=1e-16
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    #print(nc)
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
   
    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap=np.squeeze(ap)
    return p,r,ap


def my_predict(model_path,test_path):
    '''传入模型地址和测试集地址'''
    model=YOLO(model_path)
    test_path=test_path+r"\*.jpg"
    image_list=glob.glob(test_path)
    results=model(image_list)
    res_iou=[0,0,0,0]#记录总的和各个类别的iou
    res_cnt=[0,0,0,0]

    tps=[]#判断每个检测框是否是tp
    confs=[]#每个检测框的置信度
    pred_cls=[]
    target_cls=[]
    for (i,result) in enumerate(results):
        boxes=result.boxes
        txt_path=image_list[i][0:-4]+".xml"
        cls_list,box_list=read_boxes(txt_path)#当前图片真实的类别和盒子
        pred_ori=[]
        has_match=[0]*len(box_list)#判断真实框是否已经匹配
        for (j,box) in enumerate(boxes):#模型计算出来的盒子
            cls_pred=int(box.cls) 
            conf=float(box.conf)
            xyxy1=box.xyxy.tolist()
            box1=[int(t) for d2 in xyxy1 for t in d2]
            max_iou=0
            max_k=-1
        
            for (k,real_box) in enumerate(box_list):#找到跟检测目标最对应的真实框
                iou=cal_iou_xyxy(box1,real_box)
                if iou>max_iou:
                    max_iou=iou
                    max_k=k
            pred_ori.append([max_k,max_iou,conf,cls_pred,cls_list[max_k]])#检测盒子对应的真实盒子序号，最大iou，置信度，预测类别，真是类别
        pred_ori = sorted(pred_ori, key=lambda x: -x[1])#iou排序，找到每个ground_truth对应的pred
        for x in pred_ori:
            if (x[3]==x[4] and has_match[x[0]]==0 and x[1]>0.5):
                tps.append(1)
                has_match[x[0]]=1
                res_iou[0]+=x[1]
                res_iou[x[3]+1]+=x[1]
                res_cnt[0]+=1
                res_cnt[x[3]+1]+=1
            else:
                tps.append(0)
            confs.append(x[2])
            pred_cls.append(x[3])
            target_cls.append(x[4])


    res_iou=[res_iou[i]/res_cnt[i] for i in range(4)]
    tp=np.array(tps)
    tp=np.expand_dims(tp,axis=1)
    res=ap_per_class(tp,np.array(confs),np.array(pred_cls),np.array(target_cls))#计算指标
    res_pr=np.insert(res[0],0,np.mean(res[0]))
    res_r=np.insert(res[1],0,np.mean(res[1]))
    res_map=np.insert(res[2],0,np.mean(res[2]))
    show_table(res_iou,res_pr,res_r,res_map)
       
my_predict(r".\best.pt",r".\test")

