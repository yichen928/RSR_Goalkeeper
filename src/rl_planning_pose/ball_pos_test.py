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
        file_dir = './ckpts'
        self.client = roslibpy.Ros(host='192.168.50.1', port=9090) #TODO: put ip here
        self.client.run()
        self.control_points_sub = roslibpy.Topic(self.client, '/global_ball_pos', 'std_msgs/Float32MultiArray')
        self.control_points_sub.subscribe(self.ball_pos_callback)
        self.global_ball_pos = None
        self.last = time.perf_counter()


    def ball_pos_callback(self, data):
        self.global_ball_pos = data['data']
        if self.global_ball_pos is not None:
            print(self.global_ball_pos)
            curr = time.perf_counter()
            print("Ball Freq: ", 1/(curr - self.last))
            self.last = curr


def main(args):
    rpp = rl_planner_proc()
    try:
        while True:
            time.sleep(1)
            # pass
    except KeyboardInterrupt:
        rpp.client.terminate()

if __name__ == '__main__':
    main(sys.argv)