import numpy as np
import torch
from collections import deque
from motion_pred.utils.config import Config
from models.motion_pred import *
import pickle
from datetime import datetime
import os
import time

index = 0

class HumanLatent:
    def __init__(self, date_time=None):
        self.device = torch.device('cuda:0')
        self.date_time = date_time
        self.history_len = 13
        self.human_fps = 16.5
        self.human_joints = np.array([4, 5, 7, 8, 10, 11, 27, 28])  # corresponds to [right knee, ankle, foot, big toe]

        self.previous_ball = deque(maxlen=100000000)
        self.previous_ball_raw = deque(maxlen=100000000)
        self.previous_human = deque(maxlen=100000000)
        self.previous_ball_time = deque(maxlen=100000000)
        self.previous_human_time = deque(maxlen=100000000)

        cfg = Config('human_kick', test=True)

        self.model = get_kick_vae_model(cfg)
        self.model_weight = 'vae_0500.p'
        model_cp = pickle.load(open(self.model_weight, "rb"))
        self.model.load_state_dict(model_cp['model_dict'])
        self.model.eval()
        self.model.to(self.device)

        self.human_latent = None
        self.index = 0

    def get_human_latent(self):
        return self.human_latent

    def set_ball_states(self, ball_pos, ball_time):
        if len(self.previous_ball) > 0 and ball_pos[0] > 6:
            print("if", ball_pos[0], ball_time)
            self.previous_ball.append(np.copy(self.previous_ball[-1]))
            self.previous_ball_time.append(ball_time)
        else:
            print("else", ball_pos[0], ball_time)
            self.previous_ball.append(np.copy(ball_pos))
            self.previous_ball_time.append(ball_time)


        # self.previous_ball_raw.append(np.copy(ball_pos))
        # self.previous_ball.append(np.copy(ball_pos))
        # self.previous_ball_time.append(ball_time)

    def set_human_poses(self, human_pose, human_time):
        self.previous_human.append(np.copy(human_pose))
        self.previous_human_time.append(human_time)

    def run_encoding_once(self):
        print(self.human_latent)
        if len(self.previous_human_time) == 0 or len(self.previous_ball_time) == 0:
            return

        if self.previous_ball[-1][0] < 1.5:
            self.human_latent = None
            return
        if self.human_latent is not None:
            return

        human_time = self.previous_human_time[-1]
        human_idx = 1
        ball_idx = 1
        ball_pos = np.zeros((self.history_len, 3))
        human_pose = np.zeros((self.history_len, 29, 3))

        for i in range(1, self.history_len+1):
            while ball_idx + 1 < len(self.previous_ball_time) and human_time < self.previous_ball_time[-ball_idx-1]:
                ball_idx += 1

            if self.previous_ball_time[-ball_idx] - self.previous_ball_time[-ball_idx-1] == 0:
                ball_pos[-i] = self.previous_ball[-ball_idx]
            else:
                curr_ball_pos = self.previous_ball[-ball_idx] * (human_time - self.previous_ball_time[-ball_idx-1]) + self.previous_ball[-ball_idx-1] * (self.previous_ball_time[-ball_idx] - human_time)
                ball_pos[-i] = np.copy(curr_ball_pos/(self.previous_ball_time[-ball_idx]-self.previous_ball_time[-ball_idx-1]))

            while human_idx + 1 < len(self.previous_human_time) and human_time < self.previous_human_time[-human_idx-1]:
                human_idx += 1
            if self.previous_human_time[-human_idx] - self.previous_human_time[-human_idx-1] == 0:
                human_pose[-i] = self.previous_human[-human_idx]
            else:
                curr_human_pos = self.previous_human[-human_idx] * (human_time - self.previous_human_time[-human_idx-1]) + self.previous_human[-human_idx-1] * (self.previous_human_time[-human_idx] - human_time)
                human_pose[-i] = np.copy(curr_human_pos/(self.previous_human_time[-human_idx]-self.previous_human_time[-human_idx-1]))

            if human_time < self.previous_ball_time[0] or human_time < self.previous_human_time[0]:
                ball_pos[-i] = 0
                human_pose[-i] = 0
                break

            human_time = human_time - 1 / self.human_fps

        ball_pos = ball_pos[:, None, :]
        human_pose = human_pose[:, self.human_joints, :]
        human_ball = np.concatenate([human_pose, ball_pos], axis=1)
        human_ball = (human_ball[1:, :] - human_ball[:-1, :]) / 0.1

        human_ball[:2] = 0  # same with training, the first two frames are zeros

        human_ball = human_ball.reshape(self.history_len-1, 1, -1)
        human_ball = torch.from_numpy(human_ball).to(self.device).float()

        h_x = self.model.encode_x(human_ball)
        self.human_latent = h_x[0].detach().cpu().numpy().tolist()

        print(self.human_latent)

    def run_encoding_forever(self):
        while True:
            print('encoding')
            self.run_encoding_once()


if __name__ == '__main__':
    hl = HumanLatent()
    for i in range(100):
        hl.set_ball_states(np.array([i, i, i]), 0.01*i)
        hl.set_human_poses(np.ones((29, 3))*i, 1*i)

    hl.run_encoding_once()