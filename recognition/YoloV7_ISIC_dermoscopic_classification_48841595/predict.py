#predict showing example usage of the trained model. Print out any results and provides visualisations
from util.yolov7onnx.model import YOLOv7ONNX
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
directory = Path("/Users/keagancasey/Desktop/data/test/img")
images = [image for image in list(directory.glob("*.jpg"))]

def draw_box(frame, cls, x1, y1, x2, y2):

    frame_tmp = frame.copy()
    cv2.rectangle(frame_tmp, (x1,y1), (x2,y2), (255, 0, 0), 3)
    label = str(cls)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1 = max(y1, labelSize[1])
    frame_tmp = cv2.rectangle(frame_tmp, (x1, int(y1 - round(1.5*labelSize[1]))), (x1 + int(round(1.5*labelSize[0])), y1 + baseLine), (255, 255, 255), cv2.FILLED)
    frame_tmp = cv2.putText(frame_tmp, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    return frame_tmp

def draw_boxes(frame, boxes):
    h,w = frame.shape[:2]
    for box in boxes:
        frame = draw_box(
            frame,
            box.label,
            #box.score,
            int(box.x_min*w),
            int(box.y_min*h),
            int(box.x_max*w),
            int(box.y_max*h),
        )
    return frame

def put_text(img, text, coords, scale=1):
    txt_params = dict(
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        org = coords,
        fontScale = scale,
        color = (255, 172, 0),
        thickness = 1,
        lineType = cv2.LINE_AA,
    )
    cv2.putText(img, text, **txt_params)

    weights = Path("/home/keagancasey/best.onnx")
classes = {
    0: {"name":"MEL"},
    1: {"name":"NV"},
    2: {"name":"BCC"},
    3: {"name":"AKIEC"},
    4: {"name":"BKL"},
    5: {"name":"DF"},
    6: {"name":"VASC"},    
}
for image in images:
    print(image)
    try:     
        imageArr = cv2.imread(str(image))
        img_rgb = cv2.cvtColor(imageArr, cv2.COLOR_BGR2RGB)
        imageTest = YOLOv7ONNX(weights,conf_thres = 0.4, classes=classes, execution_providers=['CPUExecutionProvider'])
        boxes = imageTest.forward(img_rgb)
        newFrame = draw_boxes(img_rgb, boxes)
        plt.imshow(newFrame)
        plt.show()
    except Exception as e:
        print(e)

