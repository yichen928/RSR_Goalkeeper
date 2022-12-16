# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys, os, getopt
# from tkinter.messagebox import NO
import time

from realsense_pipeline import realsense_pipeline

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-y", "--yolo", default="/yolov7", help="Yolov7 path")
parser.add_argument("-w", "--weight", default="epoch_24.pt", help="Model weight path")
args = vars(parser.parse_args())

yolo_path = "/home/yiming-ni/yolov7"#args["yolo"]

file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(yolo_path)
sys.path.append(file_path)

import cv2
import torch
import  numpy as np
import torch

import torchvision.transforms as T

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class BallDetect:
    def __init__(self):
        self.model_weight = "/home/yiming-ni/mcSoccer_ws/src/mcSoccer_perception_planning/checkpoint.pt"#args["weight"]
        self.device = select_device('0')
        # load ObjectionDectection model
        self.net = attempt_load(self.model_weight, map_location=self.device)  # load FP32 model
        self.net.eval()
        self.classes = self.net.module.names if hasattr(self.net, 'module') else self.net.names
        self.stride = int(self.net.stride.max())  # model stride
        self.imgsz = check_img_size(360, s=self.stride)  # check img_size
        
        self.Objection_vec=[]
        self.global_pos = [0,0,0]
        # self.model_weight = "/home/mzc/code/PycharmProjects/DNN_D435/Yolo_model/yolov3.weights"
        # self.model_weight = "/media/nelson/SSD/yolov7/runs/train/exp18/weights/epoch_024.pt"
        # self.model_classname="/home/mzc/code/PycharmProjects/DNN_D435/Yolo_model/object_detection_classes_yolov3.txt"


    def detection(self,net,image):   
        # img = torch.from_numpy(np.copy(image)).cuda().squeeze().permute(2,0,1).unsqueeze(0)
        img = letterbox(image, self.imgsz, self.stride)[0]
        img = torch.from_numpy(np.copy(img)).to(self.device).squeeze().permute(2,0,1).unsqueeze(0)
        

        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        pred = net(img)[0].detach()
        # Apply NMS
        pred = non_max_suppression(pred, 0.2, 0.3)

        pos = np.zeros(4)
        max_conf = 0
        xyxy_max = None
        if len(pred[0]):
            det = pred[0]
            # Rescale boxes from img_size to image original size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            
            for *xyxy, conf, cls in reversed(det):
                # label = f'{self.classes[int(cls)]} {conf:.2f}'
                # self.Objection_vec.append(Objection(Rcet(int(xyxy[0]),int(xyxy[2]),int(xyxy[1]),int(xyxy[3])), self.classes[int(cls)],depth))
                if conf > max_conf:
                    max_conf = conf
                    pos = np.array([int(xyxy[0]),int(xyxy[2]),int(xyxy[1]),int(xyxy[3])]) # left, right, top, bottom
                    xyxy_max = xyxy
            
            # if max_conf < 0.2:
            #     pos[:] = 0.
            # else:
            plot_one_box(xyxy_max, image, color=[255,0,0], label="", line_thickness=3)
        return pos, image
    
    def prepare_detection_window(self):
        self.pipeline = realsense_pipeline()
        cv2.namedWindow('RealSenseRGB', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('RealSenseDepth', cv2.WINDOW_AUTOSIZE)
        torch.cuda.empty_cache()
       
    def run_detection_once(self):
        Position = None
        # time_hist.append(t1)
        t1 = time.perf_counter()
        image = self.pipeline.get_rgb_image()
        t2 = time.perf_counter()
        print("get image delay", (t1-t2) * 1000)
        # saving the final output 
        # as a PNG file

        # image = cv2.imread('/home/yiming-ni/Cheetah-Software-real_test/ball_video/{}.png'.format(self.idx))
        # self.idx += 1
        camera_pos, Obj_img=self.detection(self.net,image) ##检测及绘制必要的信息
        t2 = time.perf_counter()
        print("get detection", (t1-t2) * 1000)
        cv2.imshow('RealSenseRGB',Obj_img)
        # cv2.imshow('RealSenseDepth',rs_config.D435_para.depthmat)
        Keyvalue = cv2.waitKey(1)
        # if Keyvalue==27:
        #     cv2.destoryAllWindows()
        #     break
        print(f'Freq ({1/(t2 - t1):.3f} fps)')
        t2 = time.perf_counter()
        print("get image delay", (t1-t2) * 1000)
        # self.Objection_vec = []
        if not np.all(camera_pos == 0.):
            global_pos=self.pipeline.convert_actual_pos(camera_pos)
            if global_pos[0] < 7:
                self.global_pos = global_pos

    def run_detection_forever(self):
        while True:
            print('detecting')
            self.run_detection_once()

    def get_global_pos(self):
        return self.global_pos

if __name__ == '__main__':
    ball_detect = BallDetect()
    ball_detect.prepare_detection_window()
    pos = []
    last_pos = [0,0,0]
    for i in range(150):
        global_pos = ball_detect.run_detection_once()
        if global_pos is not None:
            pos.append(global_pos)
            print(np.round(global_pos, 2))
            last_pos = global_pos
        else:
            pos.append(last_pos)
            print("NONONO")

    import tkinter
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    # plotting
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d(0,3)
    ax.set_ylim3d(-1.5,1.5)
    ax.set_zlim3d(0,3)

    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines
    pos = np.array(pos)
    # print(np.round(pos, 2))
    data = [pos.T]
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    line_ani = animation.FuncAnimation(fig, update_lines, 200, fargs=(data, lines),
                                    interval=50, blit=False)

    plt.show()
 
        
