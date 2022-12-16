import sys, os
# sys.path.append('../')
import numpy as np
import time
import kinpy as kp
from kinpy.transform import Transform
try:
    from utility.utility import *
except:
    from utility import *

file_dir = os.path.dirname(os.path.abspath(__file__))

class A1FK():
    def __init__(self):
        self.a1_tf_tree = kp.build_chain_from_urdf(open(os.path.join(file_dir, "../model/a1.urdf")).read())

    def get_foot_pose(self, joint_pos, base_pos, base_rot):
        # input rot can be either quat or euler 
        root = Transform(rot=base_rot, pos=base_pos)
        th = {'FR_hip_joint':joint_pos[0], 'FR_thigh_joint':joint_pos[1], 'FR_calf_joint':joint_pos[2], 
            'FL_hip_joint':joint_pos[3], 'FL_thigh_joint':joint_pos[4], 'FL_calf_joint':joint_pos[5], 
            'RR_hip_joint':joint_pos[6], 'RR_thigh_joint':joint_pos[7], 
            'RR_calf_joint':joint_pos[8], 'RL_hip_joint':joint_pos[9], 'RL_thigh_joint':joint_pos[10], 
            'RL_calf_joint':joint_pos[11]}
        ret = self.a1_tf_tree.forward_kinematics(th,world=root)

        foot_pose = np.concatenate([ret['FR_foot'].pos, ret['FL_foot'].pos, ret['RR_foot'].pos, ret['RL_foot'].pos])
        return foot_pose
    