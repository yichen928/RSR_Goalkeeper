import logging
import random

import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.special
from six.moves import cPickle as pickle
import enum
import os

def __bernstein(t, i, n):
    """Bernstein polynom"""
    return scipy.special.binom(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier(coeff, s):
    """Calculate coordinate of a point in the bezier curve
    Coeff shape: m x n"""
    m, n = coeff.shape[0] - 1, coeff.shape[1]
    multiplier = np.stack([__bernstein(s, k, m) for k in range(m+1)]).T
    # fcn = np.matmul(multiplier, coeff)
    fcn = np.matmul(multiplier, coeff)
    return fcn

class ReferenceMotionGenerator:
    def __init__(self, filename, env_max_timesteps, secs_per_env_step):
        if not filename:
            logger.error("Error: No motion file")
            return 
                
        self.file_names = filename
        self.env_secs_per_env_step = secs_per_env_step
        
        # MotionLibrary: load data
        self.load()

        self._initMotionInterpolater()

        print("secs_per_env_step", secs_per_env_step)

        # self.randomized_bezier_coeff = np.array([[ 0.20, 0., 0.],
        #                             [ 0.15, 0.15, 0.30],
        #                             [ 0.45, 0.33, 0.51],
        #                             [ 0.25 , 0.35, 0.81],
        #                             [ 0.48, 0.4, 0.9]])

        # self.randomized_bezier_coeff = np.array([[ 0.20, 0., 0.],
        #                             [ 0.15, 0.15, 0.30],
        #                             [ 0.45, 0.33, 0.61],
        #                             [ 0.25 , 0.41, 0.71],
        #                             [ 0.48,0.56, 0.75]])

        # self.randomized_bezier_coeff = np.array([[ 0.20, 0., 0.],
        #                             [ 0.15, 0.15, 0.30],
        #                             [ 0.45, 0.33, 0.41],
        #                             [ 0.25 , 0.41, 0.31],
        #                             [ 0.48,0.56, 0.25]])

        # self.randomized_bezier_coeff = np.array([[ 0.21, 0.12, 0.],
        #                                     [ 0.21, 0.2, 0.05],
        #                                     [ 0.21, 0.3, -0.05],
        #                                     [ 0.21 , 0.4, 0.05],
        #                                     [ 0.21, .5 , 0.1]])
        # self.randomized_bezier_coeff[:,1] *= -1
        self.randomized_bezier_coeff = np.zeros(( 5, 3))
        self.squat_t = 0
        self.catch_t = 0.5

        self.coeff_in_effect = np.zeros(( 5, 3))
        self.mode_select = 0
        self.motion_phase = 0
        self.fixed_motion_duration = 0 # fixed time for the duration of the current motion
        self.motion_t = 0 # unnormalized time for the current motion
        self.motion_t_norm = 0 # normalized time (progress) for the current motion
        self.motion_t_norm_lookforward = 0 # reuse the same tensor for all lookforward to avoid malloc time(and space)

        self.time_in_sec = 0
        self.true_time_in_sec = 0
        self.init_time_in_sec = 0
        self.action_enabled = False
        
        self.period = self.motion_data["timestamp"][-1]
        
    def load(self):
        motion_data_list = []
        for filename in self.file_names:
            with open(filename, "rb") as f:
                motion_data = pickle.load(f)
            for k in motion_data.keys():
                motion_data[k] = np.array(motion_data[k])[:92]
            motion_data_list.append(motion_data)
        
        self.motion_data = {}
        self.motion_dim = {}
        for k in motion_data_list[0].keys():
            if len(motion_data_list[0][k].shape) == 2:
                self.motion_dim[k] = motion_data_list[0][k].shape[1]
                self.motion_data[k] = np.concatenate([motion_data_list[i][k] for i in range(len(motion_data_list))], axis=-1)
            else:
                self.motion_data[k] = motion_data_list[0][k]

    def _initMotionInterpolater(self):
        self.motion_data["base_pos_global"][:,:2] = self.motion_data["base_pos_global"][:,:2] - self.motion_data["base_pos_global"][0,:2]
        self.base_pos_interp_func = interpolate.interp1d(self.motion_data["timestamp"].flatten(), self.motion_data["base_pos_global"].T, kind="linear", axis=-1)
        self.base_rot_interp_func = interpolate.interp1d(self.motion_data["timestamp"].flatten(), self.motion_data["base_rot_global"].T, kind="linear", axis=-1)
        self.joint_rot_interp_func = interpolate.interp1d(self.motion_data["timestamp"].flatten(), self.motion_data["joints_rot"].T, kind="linear", axis=-1)
        self.foot_global_pos_interp_func = interpolate.interp1d(self.motion_data["timestamp"].flatten(), self.motion_data["foot_end_pos_global"].T, kind="linear", axis=-1)
        self.foot_local_pos_interp_func = interpolate.interp1d(self.motion_data["timestamp"].flatten(), self.motion_data["foot_end_pos_local"].T, kind="linear", axis=-1)

    def getBasePosition(self, look_forward=0):
        time_index = np.clip(self.time_in_sec + look_forward * self.env_secs_per_env_step, 0, self.period)
        return self.base_pos_interp_func(time_index).T

    def getBaseRotation(self, look_forward=0):
        time_index = np.clip(self.time_in_sec + look_forward * self.env_secs_per_env_step, 0, self.period)
        return self.base_rot_interp_func(time_index).T
    
    def getJointRotation(self, look_forward=0):
        time_index = np.clip(self.time_in_sec + look_forward * self.env_secs_per_env_step, 0, self.period)
        return self.joint_rot_interp_func(time_index).T
    
    def getFootGlobalPosition(self, look_forward=0):
        time_index = np.clip(self.time_in_sec + look_forward * self.env_secs_per_env_step, 0, self.period)
        return self.foot_global_pos_interp_func(time_index).T

    def getFootLocalPosition(self, look_forward=0):
        time_index = np.clip(self.time_in_sec + look_forward * self.env_secs_per_env_step, 0, self.period)
        return self.foot_local_pos_interp_func(time_index).T

    def calc_motion_t(self): # individual function as it might be different
        if self.motion_phase == 1:
            self.motion_t = self.time_in_sec - self.squat_t
        else:
            self.motion_t = 0.

    def calc_motion_t_norm(self, motion_t_norm, look_forward=0):
        if self.motion_t == 0:
            motion_t_norm = 0.
        else:
            motion_t_norm = np.clip((self.motion_t + look_forward * self.env_secs_per_env_step),\
                                np.zeros_like(self.fixed_motion_duration), self.fixed_motion_duration - 0.01) / self.fixed_motion_duration
        return motion_t_norm
        
    def get_bezier_foot_pos(self, t):
        foot_ref_residual = bezier(self.coeff_in_effect, t)
        return foot_ref_residual

    def getFootPosBezier(self, look_forward=0):
        if look_forward:
            self.motion_t_norm_lookforward = self.calc_motion_t_norm(self.motion_t_norm_lookforward, look_forward)
            return self.get_bezier_foot_pos(self.motion_t_norm_lookforward)
        return self.get_bezier_foot_pos(self.motion_t_norm)

    def getReferenceMotion(self, look_forward=0):
        ref_dict = dict()
        ref_dict["joints_rot"] = self.getJointRotation(look_forward) # periodic
        ref_dict["foot_pos_bezier"] = self.getFootPosBezier(look_forward)

        for k in ref_dict.keys():
            if 'bezier' in k:
                continue
            ref_dict[k] = ref_dict[k].reshape(-1, self.motion_dim[k])[self.mode_select - 1]

        if self.mode_select == 3:
            ref_dict["joints_rot"][:] = 0. # sidestep doesn't require this
        
        return ref_dict

    def get_motion_phase(self):
        return np.array([self.motion_phase])

    def get_bezier_coefficients(self):
        # print(self.coeff_in_effect.flatten())
        return self.coeff_in_effect.flatten()

    def get_motion_t_norm(self):
        return np.array([self.motion_t_norm])

    def get_fixed_motion_duration(self):
        return np.array([self.fixed_motion_duration])

    def set_actions_from_policy(self, actions, curr_step):
        assert len(actions) == 16, "Invalid planner action"
        self.randomized_bezier_coeff[:] = actions[:15].reshape(5,3)
        self.mode_select = int(actions[-1])

        self.update_step(curr_step)

    def set_policy(self, mode_select, curr_step):
        self.mode_select = mode_select

        self.update_step(curr_step)


    def update_step(self, curr_step):
        self.true_time_in_sec = curr_step * self.env_secs_per_env_step
        if self.true_time_in_sec == 0:
            self.action_enabled = False
            self.init_time_in_sec = 0.
        
        self.enable_action()

        if self.action_enabled:
            self.time_in_sec = self.true_time_in_sec - self.init_time_in_sec
        else:
            self.time_in_sec = 0.

        self.motion_phase = (self.time_in_sec == 0) + 2 * ((self.time_in_sec > 0) & (self.time_in_sec < self.catch_t)) + 3 * (self.time_in_sec > self.catch_t) - 1 
        
        if self.motion_phase == 1:
            self.coeff_in_effect = self.randomized_bezier_coeff[:]
            self.fixed_motion_duration = self.catch_t
        else:
            self.coeff_in_effect[:] = 0.
            self.fixed_motion_duration = 0.

        self.calc_motion_t()
        self.motion_t_norm = self.calc_motion_t_norm(self.motion_t_norm)

    def enable_action(self):
        if self.mode_select != 0 and not self.action_enabled:
            self.init_time_in_sec = self.true_time_in_sec
            self.action_enabled = True

