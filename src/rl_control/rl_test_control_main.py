#!/usr/bin/env python3

import sys, os
import time
from env import ExpEnv
import os

def main():
    try:

        file_dir = './ckpts'
        library_folder = './motion_library/'
        
        ref_file = [os.path.join(library_folder, 'a1_bump_defend_v1.4.motionlib'), os.path.join(library_folder, 'a1_bump_defend_v1.4.motionlib'), os.path.join(library_folder, 'a1_sidestep_v3.motionlib')]
        model_path = [os.path.join(file_dir, "model_750.pt"), os.path.join(file_dir, "model_660.pt"), os.path.join(file_dir, "model_300.pt")]
        # model_path = os.path.join(file_dir, "model_120.pt")

        exp = ExpEnv(ref_file=ref_file,
                    model_path=model_path,
                
                    # recv_IP="127.0.0.1",
                    # recv_port=8000,
                    # send_IP="127.0.0.1",
                    # send_port=8001)
                    recv_IP="10.0.0.60",
                    recv_port=32770,
                    send_IP="10.0.0.44",
                    send_port=32769)
            # Gazebo is running on another PC
            # recv_IP="192.168.4.155",
            # recv_port=32769,
            # send_IP="192.168.4.170",
            # send_port=32770

            # Gazebo is running on the local PC
            # recv_IP="127.0.0.1",
            # recv_port=8001,
            # send_IP="127.0.0.1",
            # send_port=8000

            # Physical exp
            # recv_IP="192.168.123.155",
            # recv_port=6006,
            # send_IP="192.168.123.247",
            # send_port=6007
        exp.run_policy()
    except KeyboardInterrupt:
        print('Restore standing! ')
        exp.pid_ctrl_restore_stand()
        # raise KeyboardInterrupt

    

if __name__ == '__main__':
    main()
    
    
