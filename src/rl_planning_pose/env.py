import torch
import numpy as np
from collections import deque
from rsl_rl.modules import ActorCritic
import time


class PlannerExpEnv():
    def __init__(self, model_path=None):

        self.rt_freq = 1000
        self.exp_env_freq = 30 
        self.num_sims_per_env_step = self.rt_freq // self.exp_env_freq
        self.secs_per_policy_step = self.num_sims_per_env_step / self.rt_freq
        self.policy_freq = 1 / self.secs_per_policy_step

        '''a1, but also applied on mini-cheetah'''                 
        self.default_target_positions = [0.0,1.0661,-2.1869, 0.0,1.0661,-2.1869, 
                                         0.0,1.0661,-2.1869, 0.0,1.0661,-2.1869]

        self.default_bezier = np.array([[[ 0.20, 0., 0.],
                                        [ 0.15, 0., 0.10],
                                        [ 0.45, 0., 0.80],
                                        [ 0.25 , 0., 0.85],
                                        [ 0.48, 0., 0.85]], 

                                        [[ 0.20, 0., 0.],
                                        [ 0.15, 0., 0.10],
                                        [ 0.45, 0., 0.30],
                                        [ 0.25 , 0., 0.35],
                                        [ 0.48, 0., 0.35]],
                                        
                                        [[ 0.21, 0., 0.],
                                        [ 0.21, 0., 0.05],
                                        [ 0.21, 0., 0.15],
                                        [ 0.21 , 0., 0.05],
                                        [ 0.21, 0. , 0.0]]])

        self.bezier_range = np.stack([
            np.array([0.05, 0.05, 0.05]),
            np.array([0.25, 0.2, 0.2]),
            np.array([0.2, 0.2, 0.1]),
            ]).T

        self.history_len = 6
        self.high_obs_act = []

        self.selected_policy = 0
        self.num_skills = 3
        self.action_dict = {}
        self.actions = np.zeros(16)
        self.actual_bezier_params = np.zeros((5,3))
        self.action_dict["bezier_params"] = np.zeros((4,3))
        self.action_dict["actual_bezier_params"] = self.actual_bezier_params
        self.action_dict["selected_policy"] = 0
        
        self.init_model(model_path)
        self.previous_robot_states = np.zeros((16+12)*self.history_len + 3)
        self.previous_ball = deque(maxlen=self.history_len) 
        self.previous_robot_pos = deque(maxlen=self.history_len) 
        
        self.ball_pos = np.zeros(3)
        self.ball_pos = np.array([2.63400006,0.39,0.095])
        self.robot_pos = np.array([0,0,0.21])
        self.human_latent = np.zeros(32)

    def init_model(self, model_path):
        # self.pi = ActorCritic( 202, 213, [12,4], actor_hidden_dims=[256, 128],
        #                     critic_hidden_dims=[256, 128])
        self.pi = ActorCritic( 234, 245, [12,4], actor_hidden_dims=[256, 128],
                            critic_hidden_dims=[256, 128])
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.pi.load_state_dict(loaded_dict['model_state_dict'])
        self.pi.eval()

    def __get_observation(self):
        
        assert not np.all(self.previous_robot_states == 0), "Robot observation hasn't connected"
        assert len(self.previous_ball) == 6, "Robot observation hasn't connected"
        
        robot_states_actions, motion_t_params = self.previous_robot_states[:168], self.previous_robot_states[168:]
        human_latent = np.array(self.human_latent).flatten()
        obj_states = np.array(self.previous_ball).flatten()
        robot_pos = np.array(self.previous_robot_pos).flatten()
        print(obj_states[-9:])
        import time
        # print(robot_pos[-3:])
        bezier_params = self.action_dict["actual_bezier_params"][1:].flatten()
        selected_policy = np.array([self.action_dict["selected_policy"]])
        self.motion_phase = motion_t_params[2]

        self.curr_obs = np.concatenate([robot_states_actions, obj_states, bezier_params, selected_policy, motion_t_params, human_latent])

    def process_send_cmd(self, motor_commands):
        return motor_commands

    def __get_action(self):

        acs = self.pi.act_inference(torch.from_numpy(self.curr_obs).to(torch.float32).unsqueeze(0))[0]
        
        acs = acs.detach().cpu().numpy()
        bezier_params = acs[:12].reshape(4,3)
        selected_policy_new = acs[-1]
        # if selected_policy_new ==1 or selected_policy_new==2:
        #     selected_policy_new = 3

        if self.motion_phase == 0:
            selected_policy = selected_policy_new
            self.selected_policy = selected_policy_new
        else:
            selected_policy = self.selected_policy

        # bezier_params = np.clip(bezier_params, -1, 1)
        bezier_params = np.tanh(bezier_params)
        bezier_params[:,1] = np.array([np.sum(bezier_params[:i+1,1]) for i in range(4)])
                
        if self.motion_phase == 0 and selected_policy == 0: # skip sudden change when action is already executed. 
            self.actual_bezier_params[:] = 0.
        else:

        # print(selected_policy)
            for i in range(self.num_skills):
                if selected_policy == i + 1:
                    self.actual_bezier_params[0] = self.default_bezier[i, 0]
                
                    if self.motion_phase == 0:
                        self.actual_bezier_params[1:] = self.default_bezier[i, 1:] + bezier_params * self.bezier_range[i]
                
                    else:
                        self.actual_bezier_params[1:] = np.clip(
                                            self.default_bezier[i, 1:] + bezier_params * self.bezier_range[i], 
                                            self.actual_bezier_params[1:] - 0.05,
                                            self.actual_bezier_params[1:] + 0.05
                                            )
            # self.actual_bezier_params[2:,2] = np.clip(self.actual_bezier_params[2:,2], 0, 0.5)
        print(self.actual_bezier_params)

        self.action_dict["actual_bezier_params"] = self.actual_bezier_params
        self.action_dict["selected_policy"] = selected_policy
        self.actions[:15] = self.actual_bezier_params.flatten()
        self.actions[-1] = selected_policy 

        self.high_obs_act.append((np.copy(self.curr_obs), np.copy(self.actions)))

    def set_robot_states(self, previous_robot_states):
        previous_robot_states = previous_robot_states[:-1]
        self.previous_robot_states[:] = previous_robot_states
        self.previous_ball.append(np.copy(self.ball_pos))
        self.previous_robot_pos.append(np.copy(self.robot_pos))

    def set_ball_states(self, ball_pos):
        self.ball_pos[:] = np.array(ball_pos)

    def set_robot_pos(self, ball_pos):
        self.robot_pos[:] = np.array(ball_pos)
    
    def get_planner_actions(self):
        return self.actions

    def set_human_latent(self, human_latent):
        self.human_latent[:] =  np.array(human_latent)

    def run_policy(self):
        self.__get_observation()
        self.__get_action()