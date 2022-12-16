#!/usr/bin/env python3

import sys, os
import time
from env import ExpEnv
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

class rl_control_proc:

    def __init__(self):
        file_dir = './ckpts'
        library_folder = './motion_library/'

        ref_file = [os.path.join(library_folder, 'a1_bump_defend_v1.4.motionlib'), os.path.join(library_folder, 'a1_sidestep_v3.motionlib')]
        model_path = [os.path.join(file_dir, "model_150.pt"), os.path.join(file_dir, "model_300.pt")]

        self.client = roslibpy.Ros(host='192.168.50.1', port=9090) #TODO: put ip here
        self.client.run()
        self.control_points_sub = roslibpy.Topic(self.client, '/control_points', 'std_msgs/Float32MultiArray')
        self.control_points_sub.subscribe(self.bezier_params_callback)
        
        self.exp = ExpEnv(ref_file=ref_file,
                    model_path=model_path,
                    # recv_IP="127.0.0.1",
                    # recv_port=8000,
                    # send_IP="127.0.0.1",
                    # send_port=8001)
                    recv_IP="10.0.0.60",
                    recv_port=32770,
                    send_IP="10.0.0.44",
                    send_port=32769)

        self.bezier_params = None

    def bezier_params_callback(self, data):
        self.bezier_params = np.array(data['data'])
        print("-----\n", self.bezier_params)
        if self.bezier_params is not None:
            self.exp.set_actions_from_policy(self.bezier_params)
            curr = time.perf_counter()
            # print("Freq: ", 1/(curr-self.last), self.global_ball_pos)
            self.last = curr
        
    def robot_state_publisher(self):
        if not self.exp.policy_running:
            return
        robot_states, robot_actions, reference_params = self.exp.get_robot_states()
        curr = time.time()
        message = np.concatenate([robot_states.flatten(), robot_actions.flatten(), reference_params.flatten(), np.array([curr])])
        self.sender.sendto(pickle.dumps(message), ("127.0.0.1", 6601))

        self.receiver.setblocking(0)
        

def main(args):
    try:
        rcp = rl_control_proc()

        rcp.exp.run_policy()
    except KeyboardInterrupt:
        import datetime
        import pickle

        ti = datetime.datetime.now()
        with open('baseline_controller_{}-{}-{}-{}.log'.format(ti.month,ti.day,ti.hour, ti.minute), 'wb') as f:
            pickle.dump(rcp.exp.low_obs_act, f)
        print('Restore standing! ')
        # rcp.pid_ctrl_restore_stand()


if __name__ == '__main__':
    main(sys.argv)
    
    