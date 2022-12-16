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

        ref_file = [os.path.join(library_folder, 'a1_bump_defend_v1.4.motionlib'), os.path.join(library_folder, 'a1_bump_defend_v1.4.motionlib'), os.path.join(library_folder, 'a1_sidestep_v3.motionlib')]
        model_path = [os.path.join(file_dir, "model_750.pt"), os.path.join(file_dir, "model_660.pt"), os.path.join(file_dir, "model_300.pt")]


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

        self.receiver = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # self.receiver.bind(('127.0.0.1', 6600))
        self.receiver.bind(('127.0.0.1', 6600))
        self.sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


    def sync_callback(self):
        if not self.exp.policy_running:
            return
        robot_states, robot_actions, reference_params = self.exp.get_robot_states()
        curr = time.time()
        message = np.concatenate([robot_states.flatten(), robot_actions.flatten(), reference_params.flatten(), np.array([curr])])
        self.sender.sendto(pickle.dumps(message), ("127.0.0.1", 6601))

        self.receiver.setblocking(0)
        ready = select.select([self.receiver], [], [], 1)
        if ready[0]:
            message = self.receiver.recv(4096)
            self.exp.set_actions_from_policy(pickle.loads(message))
        else:
            print("Receive planner action timed out!!")

def main(args):
    try:
        rcp = rl_control_proc()
        t = PT(1/30, rcp.sync_callback)
        t.start()

        rcp.exp.run_policy()
    except KeyboardInterrupt:
        print('Restore standing! ')
        import datetime
        import pickle

        ti = datetime.datetime.now()

        with open('3controller_{}-{}-{}-{}-{}.log'.format(ti.month,ti.day,ti.hour, ti.minute, ti.second), 'wb') as f:
            pickle.dump(rcp.exp.low_obs_act, f)
        # with open()
        # rcp.pid_ctrl_restore_stand()


if __name__ == '__main__':
    main(sys.argv)
    
    