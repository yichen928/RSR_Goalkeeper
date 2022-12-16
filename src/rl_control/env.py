import sys, copy

from udp_interface import udp_interface
import numpy as np
import os
# from ppo.run import train
# from baselines.common import tf_util as U
from utils.action_filter import ActionFilterButter
from utils.reference_generator import ReferenceMotionGenerator
from collections import deque
from utils.utility import *
from utils.quaternion_function import euler2quat
import time
from sshkeyboard import listen_keyboard
import threading
import torch
from rsl_rl.modules import ActorCritic
from dataclasses import dataclass, field


@dataclass
class RobotState:
    trans_vel: np.ndarray
    trans_acc: np.ndarray
    rot_quat: np.ndarray
    rot_vel: np.ndarray
    motor_pos: np.ndarray
    motor_vel: np.ndarray



NUM_MOTORS=12
class ExpEnv():
    def __init__(self, ref_file='../motions/MotionLibrary/LiftingMotion_Simulator.motionlib', model_path=None, cfg=None, 
                    recv_IP=None, recv_port=None,send_IP=None,send_port=None):

        self.robot = udp_interface(recv_IP=recv_IP,
                                recv_port=recv_port,
                                send_IP=send_IP,
                                send_port=send_port)
        self.dir = os.path.dirname(os.path.abspath(__file__))

        self.rt_freq = 1000
        self.exp_env_freq = 30 
        self.num_sims_per_env_step = self.rt_freq // self.exp_env_freq
        self.secs_per_policy_step = self.num_sims_per_env_step / self.rt_freq
        self.policy_freq = 1 / self.secs_per_policy_step

        self.motor_kps = [100.0]*12
        self.motor_kds = 4*[1.0,2.0,2.0] 

        self.motor_vel_idx = [i+6 for i in range(NUM_MOTORS)]

        '''a1, but also applied on mini-cheetah'''                 
        self.default_target_positions = [0.0,1.0661,-2.1869, 0.0,1.0661,-2.1869, 
                                         0.0,1.0661,-2.1869, 0.0,1.0661,-2.1869]

        self.action_bounds = np.array([[-0.7767,  0.7767],
                                        [-0.3011,  3.7045],
                                        [-2.8500, -0.1500],
                                        [-0.7767,  0.7767],
                                        [-0.3011,  3.7045],
                                        [-2.8500, -0.1500],
                                        [-0.7767,  0.7767],
                                        [-0.3011,  3.7045],
                                        [-2.8500, -0.1500],
                                        [-0.7767,  0.7767],
                                        [-0.3011,  3.7045],
                                        [-2.8500, -0.1500]]).T

        self.history_len = 15
        self.action_filter_order = 2
        self.action_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=self.policy_freq,order=self.action_filter_order,num_joints=NUM_MOTORS)

        self.selected_policy = 0
        self.use_planner = True
        self.ever_jump = False
        self.policy_running = False
        
        self.init_model(model_path)
        self.init_robot_state()
        self.previous_obs = deque(maxlen=self.history_len)
        self.previous_acs = deque(maxlen=self.history_len)
        self.reference_generator = ReferenceMotionGenerator(ref_file, 2000, self.secs_per_policy_step)
        self.__reset()
        if not self.use_planner:
            self.reference_generator.set_policy(self.selected_policy)

        self.low_obs_act = []
        
    def init_robot_state(self):
        trans_vel = np.zeros((3,))
        trans_acc = np.zeros((3,))
        rot_vel = np.zeros((3,))
        rot_quat = np.zeros((4,))
        motor_pos = np.zeros((NUM_MOTORS,))
        motor_vel = np.zeros((NUM_MOTORS,))
        self.obs_robot_state = RobotState(trans_vel=trans_vel, trans_acc=trans_acc, rot_quat=rot_quat, rot_vel=rot_vel, motor_pos=motor_pos, motor_vel=motor_vel)

    def init_model(self, model_path):
        self.pi = []
        if isinstance(model_path, str):
            selection = input("Single policy experiment, press 1 or 2 to select policy, any key to exit: ")
            selection = int(selection)
            if selection == 1 or selection == 2:
                self.selected_policy = selection
                self.use_planner = False
                path = [None, None]
                path[self.selected_policy - 1] = model_path
                model_path = path
            else:
                raise NotImplementedError
        for model in model_path:
            pi = ActorCritic( 499, 536, NUM_MOTORS, actor_hidden_dims=[512, 256, 128], #499, 536
                            critic_hidden_dims=[512, 256, 128])
            if model is not None:
                loaded_dict = torch.load(model, map_location=torch.device('cpu'))
                pi.load_state_dict(loaded_dict['model_state_dict'])
            pi.eval()
            self.pi.append(pi)

    def __process_recv_package(self, obs):
        self._raw_state = obs
        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        rpy = self._raw_state[0:3]
        q = euler2quat(rpy[0], rpy[1], rpy[2])
        self.obs_robot_state.motor_pos = np.array(self._raw_state[6:18])
        self.obs_robot_state.rot_quat = np.copy(np.array([q[1], q[2], q[3], q[0]]))
        # print(self.obs_robot_state.rot_quat)

        ''' Thigh and Calf joints are reversed on the real robot '''
        self.obs_robot_state.motor_pos[[1,2,4,5,7,8,10,11]] *= -1

    def __get_observation(self, acs = np.zeros(NUM_MOTORS), step = False):
        __acs = np.copy(acs)
        ref_dict_1 = self.reference_generator.getReferenceMotion(look_forward=1)
        ref_dict_4 = self.reference_generator.getReferenceMotion(look_forward=4)
        ref_dict_7 = self.reference_generator.getReferenceMotion(look_forward=7)

        ob1 = ref_dict_1["joints_rot"]
        ob4 = ref_dict_4["joints_rot"]
        ob7 = ref_dict_7["joints_rot"] 

        ob_curr = np.concatenate([self.obs_robot_state.rot_quat, self.obs_robot_state.motor_pos]) 

        bezier_param = np.concatenate([self.reference_generator.get_bezier_coefficients(), self.reference_generator.get_fixed_motion_duration(),
                                       self.reference_generator.get_motion_t_norm(), self.reference_generator.get_motion_phase()]) # motion_type: 0,1,2,3,4

        feet_pos = np.concatenate([ref_dict_1["foot_pos_bezier"], ref_dict_4["foot_pos_bezier"], ref_dict_7["foot_pos_bezier"]])
        
        if self.timestep == 0:
            [self.previous_obs.append(ob_curr) for i in range(self.history_len)]
            [self.previous_acs.append(self.default_target_positions) for i in range(self.history_len)]  
        
        ob_prev = np.concatenate([np.array(self.previous_obs).flatten(), np.array(self.previous_acs).flatten()])
        
        # print(bezier_param)
        if step:
            self.previous_obs.append(ob_curr)
            self.previous_acs.append(__acs)

        self.curr_obs = np.concatenate([ob_prev, ob_curr, ob1, ob4, ob7, bezier_param, feet_pos])

    def process_send_cmd(self, motor_commands):
        return motor_commands

    def acs_norm2actual(self, acs):
        return self.action_bounds[0] + (acs + 1)/2.0 * (self.action_bounds[1] - self.action_bounds[0])

    def acs_actual2norm(self, actual_acs):
        return (actual_acs - self.action_bounds[0])*2 / (self.action_bounds[1] - self.action_bounds[0]) - 1

    def __get_action(self):
        print(self.selected_policy, self.curr_obs[-15:-9])
        if self.selected_policy == 0:
            acs = np.copy(self.acs_actual2norm(self.default_target_positions))

        else:
            acs = self.pi[self.selected_policy - 1].act_inference(torch.from_numpy(self.curr_obs).to(torch.float32).unsqueeze(0))[0]
            acs = acs.detach().numpy()
            acs = np.clip(np.copy(acs), -1, 1)
        
        if self.selected_policy == 1 or self.selected_policy == 2:
            if self.reference_generator.time_in_sec < 0.33:
                acs[[6,9]] = np.clip(acs[[6,9]], -0.2, 0.2)
            else:
                acs[[6,9]] = np.clip(acs[[6,9]], -0.8, 0.8)
        assert acs.shape[0] == 12 and -1.0 <= acs.all() <= 1.0

        if self.timestep == 0: # prevent zero action output
            default_action = np.array(self.default_target_positions)
            self.actual_pTs_filtered = default_action
            self.action_filter.init_history(self.acs_actual2norm(default_action))

        pTs_filtered = np.copy(self.action_filter.filter(np.copy(acs)))
        actual_pTs_filtered = np.copy(self.acs_norm2actual(pTs_filtered))
        
        return actual_pTs_filtered, np.copy(self.curr_obs), np.copy(acs)

    def __env_update(self):
        # if self.timestep<3:
        self.timestep += 1
        self.time_in_sec = (self.timestep*self.num_sims_per_env_step) / self.rt_freq
        self.reference_generator.update_step(self.timestep)

    def __reset(self):
        self.action_filter.reset()
        self.timestep = 0.0
        self.est_timestep = 0
        self.time_in_sec = 0.0
        self.actual_pTs = np.zeros(NUM_MOTORS)
        self.actual_pTs_filtered = np.zeros(NUM_MOTORS)

    def pid_ctrl(self):
        policy_count = 0
        previous_time = time.time()
        t = threading.currentThread()
        a1_default_target_positions = np.array([0.0,0.9,-1.8, 0.0,0.9,-1.8, 
                                         0.0,0.9,-1.8, 0.0,0.9,-1.8])
        while getattr(t, "do_run", True):
            obs = self.robot.receive_observation()
            self.__process_recv_package(np.copy(obs))
            if policy_count % 1 == 0: 
                self.__get_observation(np.copy(self.default_target_positions), step=False)
                self.actual_pTs_filtered_sent = np.copy(a1_default_target_positions)
                for i in range(4):
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]
                cmd=self.process_send_cmd(np.concatenate((self.actual_pTs_filtered_sent,[0.0, 0.0, 0.0],np.zeros((12,)), [0.0,0.0,0.0])))
            
            self.robot.send_command(cmd)
            policy_count += 1
            current_time = time.time()
            # print("proc", "Frequency: ", 1/(current_time - previous_time + 1e-10))
            previous_time = current_time
            delay = 0.6
            _ = time.perf_counter() + delay/1000
            while time.perf_counter() < _:
                pass

    def pid_ctrl_squat_prep(self):
        policy_count = 0
        previous_time = time.time()
        t = threading.currentThread()
        a1_default_target_positions = np.array([0.0,0.9,-1.8, 0.0,0.9,-1.8, 
                                         0.0,0.9,-1.8, 0.0,0.9,-1.8])
        
        while getattr(t, "do_run", True):
            obs = self.robot.receive_observation()
            self.__process_recv_package(np.copy(obs))
            if policy_count > 30: 
                self.actual_pTs_filtered_sent = np.copy(self.default_target_positions)
                for i in range(4):
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]
                break
            
            else:
                self.__get_observation(np.copy(self.default_target_positions), step=False)
                self.actual_pTs_filtered_sent = (30 - policy_count) / 30 * a1_default_target_positions + policy_count / 30 * np.array(self.default_target_positions)
                for i in range(4):
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]
                cmd=self.process_send_cmd(np.concatenate((self.actual_pTs_filtered_sent,[0.0, 0.0, 0.0],np.zeros((12,)), [0.0,0.0,0.0])))
                
                self.robot.send_command(cmd)
                policy_count += 1
                current_time = time.time()
                # print("proc", "Frequency: ", 1/(current_time - previous_time + 1e-10))
                previous_time = current_time
                delay = 0.6
                _ = time.perf_counter() + delay/1000
                while time.perf_counter() < _:
                    pass
    
    def press(self, key):
        print("Doing nothing")

    def set_actions_from_policy(self, planner_actions=None):
        import time
        # print("delay: ", time.time()-planner_actions[-1])
        planner_actions = planner_actions[:-1]
        
        if not self.use_planner:
            raise NotImplementedError
        else:
            if not self.reference_generator.action_enabled:
                self.selected_policy = int(planner_actions[-1])

        if self.selected_policy == 1:
            self.ever_jump = True
        
        # if self.reference_generator.motion_phase == 2:
        #     if self.ever_jump:
        #         self.selected_policy = 1
        #     else:
        #         self.selected_policy = 2
        
        planner_actions[-1] = self.selected_policy
        
        self.reference_generator.set_actions_from_policy(planner_actions, self.timestep)

    def get_robot_states(self, planner_actions=None):
        ob_curr = np.concatenate([self.obs_robot_state.rot_quat, self.obs_robot_state.motor_pos]) 
        robot_states = np.concatenate([np.array(self.previous_obs)[-5:].flatten(), ob_curr])
        robot_actions = np.concatenate([np.array(self.previous_acs)[-5:].flatten(), self.actual_pTs_filtered])
        reference_params = np.concatenate([self.reference_generator.get_fixed_motion_duration(), 
                                    self.reference_generator.get_motion_t_norm(), self.reference_generator.get_motion_phase()])
        return robot_states, robot_actions, reference_params
    
    def run_policy(self):
        proc = threading.Thread(target=self.pid_ctrl)
        proc.start()

        listen_keyboard(on_press=self.press, until='space')
        proc.do_run = False
        previous_time = time.perf_counter()

        # proc_squat_prep = threading.Thread(target=self.pid_ctrl_squat_prep)
        # proc_squat_prep.start()

        # listen_keyboard(on_press=self.press, until='space')
        # proc_squat_prep.do_run = False
        self.pid_ctrl_squat_prep()

        while(True):
            if not self.policy_running:
                self.policy_running = True
            obs = self.robot.receive_observation()
            self.__process_recv_package(obs)
            if self.est_timestep % 1 == 0:
                # print("self.est_timestep", self.est_timestep)
                # print("self.num_sims_per_env_step", self.num_sims_per_env_step)
                
                # if self.timestep < 30:
                #     self.reference_generator.set_policy(0, self.timestep)
                #     self.selected_policy = 0
                # else:
                #     self.reference_generator.set_policy(1, self.timestep)
                #     self.selected_policy = 1
                
                if self.timestep == 0:
                    self.actual_pTs_filtered = np.zeros(12)
                self.__get_observation(np.copy(self.actual_pTs_filtered), step=True)
                self.actual_pTs_filtered, ob, ac= self.__get_action()
                
                self.low_obs_act.append((np.copy(self.curr_obs), np.copy(self.actual_pTs_filtered)))

                self.actual_pTs_filtered = np.round(self.actual_pTs_filtered,5)

                self.actual_pTs_filtered_sent = np.copy(self.actual_pTs_filtered)

                for i in range(4):
                    # self.actual_pTs_filtered_sent[3*i] = -self.actual_pTs_filtered_sent[3*i]
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]

                self.robot.send_command(self.actual_pTs_filtered_sent)
            
                self.est_timestep = 0
                self.__env_update()

            else:
                # send previous action package 
                self.robot.send_command(self.actual_pTs_filtered_sent)
                time.sleep(0.00001)

            current_time = time.time()
            # print("Frequency: ", 1/(current_time - previous_time))
            previous_time = current_time
            self.est_timestep += 1


    def pid_ctrl_restore_stand(self):
        policy_count = 0
        a1_default_target_positions = np.array([0.0,0.9,-1.8, 0.0,0.9,-1.8, 
                                         0.0,0.9,-1.8, 0.0,0.9,-1.8])
        obs = self.robot.receive_observation()
        self.__process_recv_package(np.copy(obs))
        landing_joint_pos = np.array(obs[6:18])
        landing_joint_pos[[1,2,4,5,7,8,10,11]] *= -1
        restore_duration = 60
        while policy_count < 60:
            obs = self.robot.receive_observation()
            self.__process_recv_package(np.copy(obs))
            if policy_count > restore_duration: 
                self.actual_pTs_filtered_sent = np.copy(a1_default_target_positions)
                for i in range(4):
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]
                cmd=self.process_send_cmd(np.concatenate((self.actual_pTs_filtered_sent,[0.0, 0.0, 0.0],np.zeros((12,)), [0.0,0.0,0.0])))
            
            else:
                self.actual_pTs_filtered_sent = (restore_duration - policy_count) / restore_duration * landing_joint_pos + policy_count / restore_duration * np.array(a1_default_target_positions)
                for i in range(4):
                    self.actual_pTs_filtered_sent[3*i+1] = -self.actual_pTs_filtered_sent[3*i+1]
                    self.actual_pTs_filtered_sent[3*i+2] = -self.actual_pTs_filtered_sent[3*i+2]
                cmd=self.process_send_cmd(np.concatenate((self.actual_pTs_filtered_sent,[0.0, 0.0, 0.0],np.zeros((12,)), [0.0,0.0,0.0])))
            
            self.robot.send_command(cmd)
            policy_count += 1
            current_time = time.time()
            # print("proc", "Frequency: ", 1/(current_time - previous_time + 1e-10))
            previous_time = current_time
            delay = 0.6
            _ = time.perf_counter() + delay/1000
            while time.perf_counter() < _:
                pass