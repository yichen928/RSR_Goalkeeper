import roslibpy
import numpy as np
import sys
import os
import cv2
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
# from ball_detect import BallDetect
from ball_detect_v7 import BallDetect
import base64
import logging
import time
from threading import Timer, Thread, Event
from copy import deepcopy


class ball_detect_proc:

    def __init__(self):

        self.client = roslibpy.Ros(host='192.168.50.20', port=9090)
        self.client.run()
        # self.talker = roslibpy.Topic(self.client, '/camera_ball_pos', 'std_msgs/Float32MultiArray')
        self.time_talker = roslibpy.Topic(self.client, '/camera_ball_pos_with_time', 'std_msgs/Float32MultiArray')

        # self.listener.subscribe(self.callback)
        # self.listener_depth.subscribe(self.depth_callback)
        self.detector = BallDetect()
        self.detector.prepare_detection_window()
        # cv2.namedWindow('rgb',cv2.WINDOW_AUTOSIZE)
        self.img = None
        self.depth = None
    
    def detect(self):
        ball_pos = self.detector.get_global_pos()
        if ball_pos is None:
            print("No ball detected")
            return
        # msg = roslibpy.Message({'data': ball_pos[0]})
        # self.talker.publish(msg)

        ball_pose_time = deepcopy(ball_pos[0])
        ball_pose_time.append(ball_pos[1])
        time_msg = roslibpy.Message({'data': ball_pose_time})
        self.time_talker.publish(time_msg)

        print(ball_pose_time)

    def run_detector(self):
        self.detector.run_detection_forever()

def main(args):
    class PT():

        def __init__(self, t, hFunction):
            self.t = t
            self.hFunction = hFunction
            self.thread = Timer(self.t, self.handle_function)

        def handle_function(self):
            self.hFunction()
            self.thread = Timer(self.t, self.handle_function)
            self.thread.start()

        def start(self):
            self.thread.start()

    bdp = ball_detect_proc()
    t = PT(1/10, bdp.detect)
    t.start()
    try:
        while bdp.client.is_connected:
            bdp.detector.run_detection_once()
            # pass
    except KeyboardInterrupt:
        bdp.time_talker.unadvertise()
        bdp.client.terminate()

if __name__ == '__main__':
    main(sys.argv)
