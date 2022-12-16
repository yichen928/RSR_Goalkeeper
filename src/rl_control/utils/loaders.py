import os
import json
import logging

import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

from utility import Quat2Rxyz
from a1_fk import A1FK

logger = logging.Logger("MocapDataLoader", level=logging.INFO)


# FR hip, thigh, calf, 
# FL hip, thigh, calf, 
# RR hip, thigh, calf, 
# RL hip, thigh, calf
class AnimLoader():
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        #self.library_folder = self.dir + '/../motions/MotionLibrary/'
        #self.motion_file = 'reference_motion_putdown.csv'
        #self.saved_filename = 'PuttingDownMotion_Simulator.motionlib'
        self.saved_dict = dict()
        self.a1_fk = A1FK()
        self.base_pos_idx = [2,3,4]
        self.base_rot_idx = [5,6,7]
        self.motor_idx = [i for i in range(12)]

    def loadFromCsv(self, filename):
        self.anim_data = pd.read_csv(filename).to_numpy()
        self.time = self.anim_data[:, 1] - self.anim_data[0,1]
        self.time = self.time.reshape((self.time.shape[0],1))
        self.base_pos_list = self.anim_data[:, self.base_pos_idx]
        self.base_pos_list[:,0] = self.base_pos_list[:,0] - self.base_pos_list[0,0]
        self.base_pos_list[:,1] = self.base_pos_list[:,1] - self.base_pos_list[0,1]
        self.base_rot_list = self.anim_data[:, self.base_rot_idx]  # pitch y; roll x; yaw z
        self.base_rot_list[:, 2] = self.base_rot_list[:, 2] + np.deg2rad(90)
        joints = self.anim_data[:, 8:]
        # print(joints[0])
        
        self.motor_pos_list = joints[:, self.motor_idx]
        # inverse blender joint pos direction
        self.motor_pos_list[:,1] = -self.motor_pos_list[:,1]
        self.motor_pos_list[:,6] = -self.motor_pos_list[:,6]
        self.motor_pos_list[:,7] = -self.motor_pos_list[:,7]
        self.motor_pos_list[:,9] = -self.motor_pos_list[:,9]

        self.joint_pos_list = np.copy(joints)
        self.__get_footpos()
        self.__pack_dict_to_save()

    def __get_footpos(self):
        self.foot_pose_list_relative = np.zeros((len(self.time),6))
        self.foot_pose_list_absolute = np.zeros((len(self.time),6))
        for idx, t in enumerate(self.time):
            ref_base_pos = self.base_pos_list[idx,:]
            ref_base_rot = self.base_rot_list[idx,:]
            # print(ref_base_pos)
            ref_jnt = self.motor_pos_list[idx, :]

            foot_pose_relative = self.a1_fk.get_foot_pose(joint_pos=ref_jnt, base_pos=[0,0,0], base_rot=[0,0,0])
            foot_pose_absolute = self.a1_fk.get_foot_pose(joint_pos=ref_jnt, base_pos=ref_base_pos, base_rot=ref_base_rot)
            self.foot_pose_list_relative[idx,:] = foot_pose_relative
            self.foot_pose_list_absolute[idx,:] = foot_pose_absolute
            print(foot_pose_absolute[0:6])

    def __pack_dict_to_save(self):
        self.saved_dict['Time'] = self.time
        self.saved_dict['Base_Pos'] = self.base_pos_list
        self.saved_dict['Base_Rot'] = self.base_rot_list
        self.saved_dict['Motor_Pos'] = self.motor_pos_list
        self.saved_dict['Foot_Pose_Relative'] = self.foot_pose_list_relative
        self.saved_dict['Foot_Pose_Absolute'] = self.foot_pose_list_absolute
        
    #@staticmethod
    def saveAs(self):
        filename = self.library_folder+self.saved_filename
        with open(filename, "wb") as f:
            pickle.dump(self.saved_dict, f)
        
        logger.info("File saved as \"{0}\"".format(filename))

# if __name__ == "__main__":
#     al = AnimLoader()
#     al.load_anim_data()
#     al.saveAs()

# from a1_fk import A1FK

# FR hip, thigh, calf, 
# FL hip, thigh, calf, 
# RR hip, thigh, calf, 
# RL hip, thigh, calf

"""
data format:
timestamp base_pos_global base_rot_global joints_rot base_vel_global joints_vel foot_end_pos_global foot_end_pos_local foot_end_vel_global
[1]       [3]             [4]             [12]       [3]             [12]       [12]                [12]               [12]

joint_rot:
FR  FL  RR  RL
[3] [3] [3] [3]

hip, thigh, knee

Store data format:
timestamp base_pos_global base_rot_global joints_rot base_vel_global joints_vel foot_end_pos_global foot_end_pos_local foot_end_vel_global
[1]       [3]             [3]             [12]       [3]             [12]       [12]                [12]               [12]
"""

class MocapDataLoader():
    LIBRARY_PATH = "../motion_library/"

    def __init__(self):
        self.a1_fk = A1FK()

    # @staticmethod
    def loadFromCsv(self, filename):

        logger.info("Reading data from \"{0}\"...".format(filename))
        with open(filename, "r", encoding="utf-8") as f:
            buffer = f.read()
        

        data = json.loads(buffer)

        duration_per_frame = data.get("FrameDuration")
        frames = data.get("Frames")

        logger.info("File loaded:\n - duration: {duration}\n - frames: {n_frames}".format(duration=duration_per_frame*len(frames), n_frames=len(frames)))

        ref_motion = {
            "timestamp": [],
            "base_pos_global": [],
            "base_rot_global": [],
            "joints_rot": [],
            "base_vel_global": [],
            "joints_vel": [],
            "foot_end_pos_global": [],
            "foot_end_pos_local": [],
            "foot_end_vel_global": [],
        }
        
        # frames.reverse()

        # here f stands for frame
        for i, f in enumerate(frames):
            
            base_position = np.array([f[0], 0.0, f[2]], dtype=np.float32)
            #body_position_sequence[:,0] = base_pos_list[:,0] - base_pos_list[0,0]
            #body_position_sequence[:,1] = self.base_pos_list[:,1] - self.base_pos_list[0,1]
            
            base_rotation = Quat2Rxyz(np.array([f[3], f[4], f[5], f[6]]))  # pitch y; roll x; yaw z
            base_rotation = np.array([base_rotation[1], base_rotation[0], base_rotation[2]-np.pi], dtype=np.float32)
            #body_rotation[2] = base_rot_list[2] + np.deg2rad(90)

            # extract joint segment from array
            j = f[7:]
            joint_rotation = np.array([
                j[0], j[1], j[2], 
                j[3], j[4], j[5], 
                j[6], j[7], j[8], 
                j[9], j[10], j[11]
                ], dtype=np.float32)

            foot_pose_relative = np.array(self.a1_fk.get_foot_pose(joint_pos=joint_rotation, base_pos=[0,0,0], base_rot=[0,0,0]))
            foot_pose_absolute = np.array(self.a1_fk.get_foot_pose(joint_pos=joint_rotation, base_pos=base_position, base_rot=base_rotation))
            
            # # axis remapping
            # joint_rotation[1] += np.deg2rad(90)
            # joint_rotation[2] *= -1
            # joint_rotation[4] += np.deg2rad(90)
            # joint_rotation[5] *= -1
            # joint_rotation[7] += np.deg2rad(90)
            # joint_rotation[8] *= -1
            # joint_rotation[10] += np.deg2rad(90)
            # joint_rotation[11] *= -1

            # # Manually set abductions to 0
            # joint_rotation[0] = 0
            # joint_rotation[3] = 0
            # joint_rotation[6] = 0
            # joint_rotation[9] = 0

            ref_motion["timestamp"].append(i * duration_per_frame)
            ref_motion["base_pos_global"].append(base_position)
            ref_motion["base_rot_global"].append(base_rotation)
            ref_motion["joints_rot"].append(joint_rotation)
            ref_motion["joints_vel"].append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            ref_motion["foot_end_pos_global"].append(foot_pose_absolute)
            ref_motion["foot_end_pos_local"].append(foot_pose_relative)
            ref_motion["foot_end_vel_global"].append(np.array([0, 0, 0], dtype=np.float32))

        
        ref_motion["timestamp"] = np.array(ref_motion["timestamp"])
        ref_motion["base_pos_global"] = np.array(ref_motion["base_pos_global"])
        ref_motion["base_rot_global"] = np.array(ref_motion["base_rot_global"])
        ref_motion["joints_rot"] = np.array(ref_motion["joints_rot"])
        ref_motion["base_vel_global"] = np.array(ref_motion["base_vel_global"])
        ref_motion["joints_vel"] = np.array(ref_motion["joints_vel"])
        ref_motion["foot_end_pos_global"] = np.array(ref_motion["foot_end_pos_global"])
        ref_motion["foot_end_pos_local"] = np.array(ref_motion["foot_end_pos_local"])
        ref_motion["foot_end_vel_global"] = np.array(ref_motion["foot_end_vel_global"])

        print(ref_motion["base_pos_global"])

        return ref_motion

    def _resolveFootIKPosition(self):
        self.foot_pose_list_relative = np.zeros((len(self.time),6))
        self.foot_pose_list_absolute = np.zeros((len(self.time),6))
        for idx, t in enumerate(self.time):
            ref_base_pos = self.base_pos_list[idx,:]
            ref_base_rot = self.base_rot_list[idx,:]
            # print(ref_base_pos)
            ref_jnt = self.motor_pos_list[idx, :]

            foot_pose_relative = self.a1_fk.get_foot_pose(joint_pos=ref_jnt, base_pos=[0,0,0], base_rot=[0,0,0])
            foot_pose_absolute = self.a1_fk.get_foot_pose(joint_pos=ref_jnt, base_pos=ref_base_pos, base_rot=ref_base_rot)
            self.foot_pose_list_relative[idx,:] = foot_pose_relative
            self.foot_pose_list_absolute[idx,:] = foot_pose_absolute
            print(foot_pose_absolute[0:6])

    @staticmethod
    def saveAs(filename, data):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
        logger.info("File saved as \"{0}\"".format(filename))

    # @staticmethod
    def parse(filename):
        data_obj = MocapDataLoader()
        data = data_obj.loadFromCsv(filename)
        target_filename = filename[:filename.rfind(".")] + ".motionlib"
        MocapDataLoader.saveAs(target_filename, data)

if __name__ == "__main__":
    file = "../motion_library/a1_trot.txt"
    
    MocapDataLoader.parse(file)
    # loader = AnimLoader()
    # loader.loadFromCsv(file)
    # loader.saveAs()
