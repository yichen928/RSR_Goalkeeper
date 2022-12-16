#!/usr/bin/env python3

import sys, os
import time
from env import PlannerExpEnv
import os
from threading import Timer, Thread, Event
import socket
import pickle
import select
import numpy as np
import roslibpy

class PT():

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)
        self.last = 0.

    def handle_function(self):
        self.thread = Timer(self.t, self.handle_function)
        # curr = time.perf_counter()
        # print("Freq", 1 / (curr-self.last))
        # self.last = curr
        self.hFunction()
        self.thread.start()

    def start(self):
        self.thread.start()

class rl_planner_proc:

    def __init__(self):
        # file_dir = './ckpts'
        file_dir = './ckpts'

        model_path = os.path.join(file_dir, "model_19200.pt")

        self.exp = PlannerExpEnv(model_path=model_path)

        self.receiver = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.receiver.bind(('127.0.0.1', 6601))

        # self.talker = roslibpy.Topic(self.client, '/control_points', 'std_msgs/Float32MultiArray')

        self.sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        self.sync_timestep = 0

    def sync_callback(self):
        self.receiver.setblocking(0)

        ready = select.select([self.receiver], [], [], 1)
        if ready[0]:
            message = self.receiver.recv(4096)
            self.exp.set_robot_states(pickle.loads(message))
            self.sync_timestep += 1
            if self.sync_timestep > 12:
                if self.sync_timestep % 3 == 0:
                    self.exp.run_policy()
            # message = self.exp.get_planner_actions()
            a = np.array([[ 0.20, 0., 0.],
                                    [ 0.15, -0.15, 0.30],
                                    [ 0.45, -0.3, 0.5],
                                    [ 0.25 , -0.45, 0.15],
                                    [ 0.48, -0.50, 0.40]])
            if self.sync_timestep > 100:
                message = np.concatenate([a.flatten(), np.array([2])])
            else:
                message = np.concatenate([a.flatten(), np.array([0])])
            import time
            message = np.append(message, time.time())
            self.sender.sendto(pickle.dumps(message), ("127.0.0.1", 6600))
        else:
            print("Receive observation timed out!!")
        

    def ball_pos_callback(self, data):
        self.exp.set_ball_states(data)

    def publish_bezier_params(self):
        bezier_params = self.exp.get_planner_actions()
        msg = roslibpy.Message({'data': bezier_params})
        self.talker.publish(msg)

def main(args):
    
    rpp = rl_planner_proc()
    t = PT(1/30, rpp.sync_callback)
    t.start()
    # t2 = PT(1/10, rpp.publish_bezier_params)
    # t2.start()

    curr = time.perf_counter()
    while True:
        time.sleep(1)
        if (time.perf_counter() - curr) > 10:
            for i in range(15):
                pos = np.array([2.63400006,0.7,0.095]) - np.array([0,0.3,0.1]) * (15-i) / 15
                rpp.ball_pos_callback(pos)
                time.sleep(1/30)

if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print('Planner closing')

