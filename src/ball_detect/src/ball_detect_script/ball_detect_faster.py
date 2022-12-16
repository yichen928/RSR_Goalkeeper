# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys, os, getopt
# from tkinter.messagebox import NO
import time

from realsense_pipeline import realsense_pipeline

file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_path)

import cv2
import torch
import  numpy as np

class BallDetect:
    def __init__(self):
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom')  # or yolov5n - yolov5x6, custom
        self.model = torch.hub.load('my_yolov5', 'custom', 'my_yolov5s.engine', source='local', force_reload=True)  # or yolov5n - yolov5x6, custom
        self.model.conf = 0.2
        self.model.iou = 0.4
        self.model.max_det = 1

        
        self.Objection_vec=[]
        self.global_pos = None
        # self.model_weight = "/home/mzc/code/PycharmProjects/DNN_D435/Yolo_model/yolov3.weights"
        # self.model_weight = "/media/nelson/SSD/yolov7/runs/train/exp18/weights/epoch_024.pt"
        # self.model_classname="/home/mzc/code/PycharmProjects/DNN_D435/Yolo_model/object_detection_classes_yolov3.txt"


    def detection(self,image):
        t1 = time.time()
        results = self.model(image, size=640)
        print(results)
        # results.print()
        try:
            box = results.xyxy[0][0].int().tolist()
            # image = cv2.rectangle(image, box[:2], box[2:4], (255, 0, 0), 2)
            pos = np.array([box[0], box[2], box[1], box[3]])
            t2 = time.time()
            # print(1/(t2-t1))
            return pos, image
        except:
            print('No Object')
            return np.array([0,0,0,0]), image


    def prepare_detection_window(self):
        self.pipeline = realsense_pipeline()

        # cv2.namedWindow('RealSenseRGB', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('RealSenseDepth', cv2.WINDOW_AUTOSIZE)
        torch.cuda.empty_cache()
       
    def run_detection_once(self):
        Position = None
        image = self.pipeline.get_rgb_image()
        cv2.imwrite('color_img.png', image)
        t0 = time.time()
        camera_pos, Obj_img=self.detection(image) ##检测及绘制必要的信息

        if not np.all(camera_pos == 0.):
            global_pos = self.pipeline.convert_actual_pos(camera_pos)
            self.global_pos = (global_pos, t0)

    def run_detection_forever(self):
        while True:
            print('detecting')
            self.run_detection_once()

    def get_global_pos(self):
        return self.global_pos


if __name__ == "__main__":
    ball_detact = BallDetect()
    img = cv2.imread("color_image.png")
    _, res_img = ball_detact.detection(img)
    cv2.imwrite("res_image.png", res_img)