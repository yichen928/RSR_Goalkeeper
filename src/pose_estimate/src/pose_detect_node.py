import roslibpy
import numpy as np
import sys
import os
import cv2

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
from pose_estimate import PoseEstimate
import base64
import logging
import time
from threading import Timer, Thread, Event


class pose_estimate_proc:

    def __init__(self):
        self.client = roslibpy.Ros(host='192.168.50.20', port=9090)
        self.client.run()

        # self.talker = roslibpy.Topic(self.client, '/human_pose', 'std_msgs/Float32MultiArray')
        self.time_talker = roslibpy.Topic(self.client, '/human_pose_with_time', 'std_msgs/Float32MultiArray')

        self.detector = PoseEstimate()
        self.detector.prepare_detection_window()

        self.img = None
        self.depth = None

    def detect(self):
        human_pose = self.detector.get_global_pos()
        if human_pose is None:
            print("No human detected")
            return
        else:
            t1 = human_pose[1]
            human_pose = np.copy(human_pose[0]).flatten().tolist()
            # msg = roslibpy.Message({'data': human_pose})
            # self.talker.publish(msg)
            human_pose.append(t1)
            time_msg = roslibpy.Message({'data': human_pose})
            print(human_pose)
            self.time_talker.publish(time_msg)

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

    pep = pose_estimate_proc()
    t = PT(1 / 10, pep.detect)
    t.start()
    try:
        while pep.client.is_connected:
            pep.detector.run_detection_once()
            # pass
    except KeyboardInterrupt:
        pep.time_talker.unadvertise()
        pep.client.terminate()


if __name__ == '__main__':
    client = roslibpy.Ros(host='192.168.50.20', port=9090)
    client.run()
    main(sys.argv)
