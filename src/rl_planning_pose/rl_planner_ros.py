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
        self.client = roslibpy.Ros(host='localhost', port=9090) #TODO: put ip here
        self.client.run()
        self.control_points_sub = roslibpy.Topic(self.client, '/global_ball_pos_with_time', 'std_msgs/Float32MultiArray')
        self.control_points_sub.subscribe(self.ball_pos_callback)

        self.human_latent_sub = roslibpy.Topic(self.client, '/human_latent', 'std_msgs/Float32MultiArray')
        self.human_latent_sub.subscribe(self.human_latent_callback)
        
        # self.control_points_sub2 = roslibpy.Topic(self.client, '/vrpn_client_node/cheetah_body/odometry/mocap', 'nav_msgs/Odometry')
        # self.control_points_sub2.subscribe(self.robot_pos_callback)
        self.global_ball_pos = None
        self.global_robot_pos = np.zeros(3)

        self.talker = roslibpy.Topic(self.client, '/control_points', 'std_msgs/Float32MultiArray')

        model_path = os.path.join(file_dir, "model_10000.pt")

        self.exp = PlannerExpEnv(model_path=model_path)

        self.receiver = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.receiver.bind(('127.0.0.1', 6601))

        self.sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        self.sync_timestep = 0

        self.last = time.perf_counter()
    
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
            message = self.exp.get_planner_actions()
            import time
            message = np.append(message, time.time())
            self.sender.sendto(pickle.dumps(message), ("127.0.0.1", 6600))    
        else:
            print("Receive observation timed out!!")
        
    def robot_pos_callback(self, data):
        self.global_robot_pos = np.array([data["pose"]["pose"]["position"]["y"], -data["pose"]["pose"]["position"]["x"], data["pose"]["pose"]["position"]["z"]])

    def human_latent_callback(self, data):
        self.human_latent = data['data']
        if self.human_latent is not None:
            self.exp.set_human_latent(self.human_latent)
            curr = time.perf_counter()
            self.last = curr

    def ball_pos_callback(self, data):
        # self.global_ball_pos = np.array([data["pose"]["pose"]["position"]["y"], -data["pose"]["pose"]["position"]["x"], data["pose"]["pose"]["position"]["z"]])
        self.global_ball_pos = data['data'][:-1]
        # self.global_ball_pos[:2] = self.global_ball_pos[:2] - self.global_robot_pos[:2]
        # print("rec")
        if self.global_ball_pos is not None:
            # print("ball", self.global_ball_pos)
            self.exp.set_ball_states(self.global_ball_pos)
            curr = time.perf_counter()
            # print("Freq: ", 1/(curr-self.last), self.global_ball_pos)
            self.last = curr

    def publish_bezier_params(self):
        bezier_params = self.exp.get_planner_actions()
        bezier_params = bezier_params.tolist()
        msg = roslibpy.Message({'data': bezier_params})
        self.talker.publish(msg)



def main(args):
    
    rpp = rl_planner_proc()
    t = PT(1/30, rpp.sync_callback)
    t.start()
    t2 = PT(1/10, rpp.publish_bezier_params)
    t2.start()
    try:
        while True:
            time.sleep(10)
            # pass
    except KeyboardInterrupt:
        import datetime
        import pickle

        ti = datetime.datetime.now()

        with open('demoplanner_{}-{}-{}-{}-{}.log'.format(ti.month,ti.day,ti.hour, ti.minute, ti.second), 'wb') as f:
            pickle.dump(rpp.exp.high_obs_act, f)

        rpp.client.terminate()
        raise Exception
        

if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        
        print('Planner closing')