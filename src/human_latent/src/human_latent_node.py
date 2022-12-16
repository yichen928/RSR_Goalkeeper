import roslibpy
import numpy as np
import sys
import os
import cv2
import pickle

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
from human_latent import HumanLatent
# from ball_detect_faster import BallDetect
import base64
import logging
import time
from threading import Timer, Thread, Event
from _datetime import datetime

now = datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
height = 0.44

class human_latent_encode_proc:

    def __init__(self):
        self.client = roslibpy.Ros(host='localhost', port=9090)
        self.client.run()

        self.encoder = HumanLatent(date_time)

        self.ball_sub = roslibpy.Topic(self.client, '/camera_ball_pos_with_time', 'std_msgs/Float32MultiArray')
        self.ball_sub.subscribe(self.ball_pos_callback)

        self.human_sub = roslibpy.Topic(self.client, '/human_pose_with_time', 'std_msgs/Float32MultiArray')
        self.human_sub.subscribe(self.human_pos_callback)

        self.talker = roslibpy.Topic(self.client, '/human_latent', 'std_msgs/Float32MultiArray')

    def encode(self):
        human_latent = self.encoder.get_human_latent()
        if human_latent is None:
            print("No human latent")
            return
        print(human_latent, time.perf_counter())
        msg = roslibpy.Message({'data': human_latent})
        self.talker.publish(msg)

    def run_encoder(self):
        self.encoder.run_encoding_forever()

    def run_encoder_once(self):
        self.encoder.run_encoding_once()

    def ball_pos_callback(self, data):
        self.global_ball_pos = data['data']
        if self.global_ball_pos is not None:
            global_ball_pose = np.array(self.global_ball_pos[:-1])
            ball_time = self.global_ball_pos[-1]
            global_ball_pose[-1] = global_ball_pose[-1] + height
            self.encoder.set_ball_states(global_ball_pose, ball_time)

        # os.makedirs(date_time, exist_ok=True)
        # with open("%s/ball_pos.pkl"%date_time, "wb") as file:
        #     pickle.dump((self.encoder.previous_ball, self.encoder.previous_ball_raw, self.encoder.previous_ball_time), file)

    def human_pos_callback(self, data):
        self.global_human_pose = data['data']
        if self.global_human_pose is not None:
            global_human_pose = np.array(self.global_human_pose[:-1]).reshape(29, 3)
            human_time = self.global_human_pose[-1]

            self.encoder.set_human_poses(global_human_pose, human_time)

            self.run_encoder_once()

            # os.makedirs(date_time, exist_ok=True)
            # with open("%s/human_pose.pkl" % date_time, "wb") as file:
            #     pickle.dump((self.encoder.previous_human, self.encoder.previous_human_time), file)
            #
            # if self.encoder.human_latent is not None:
            #     with open("%s/human_latent.pkl" % date_time, "wb") as file:
            #         pickle.dump(self.encoder.human_latent, file)

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

    hlep = human_latent_encode_proc()
    t = PT(1 / 10, hlep.encode)
    t.start()
    # try:
    #     while hlep.client.is_connected:
    #         hlep.encoder.run_encoding_once()
    #         # pass
    # except KeyboardInterrupt:
    #     hlep.talker.unadvertise()
    #     hlep.client.terminate()


if __name__ == '__main__':
    main(sys.argv)
