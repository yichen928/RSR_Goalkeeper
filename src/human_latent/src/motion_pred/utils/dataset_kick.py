import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader

device = None
def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

class DatasetKick(torch.utils.data.Dataset):
    '''
    Human Kicking ball dataset
    '''
    def __init__(self, mode="train", if_residual=False):
        self.mode = mode
        self.std, self.mean = None, None
        self.normalized = None # if the data is normalized so far
        # iterator specific
        # self.sample_ind = None
        self.if_residual = if_residual
        # self.traj_dim =    # To be determined
        # NOTE: if actual timesteps is smaller than predefined timesteps, 
        # the data will be filled with some default values
        self.x_numsteps = 12  # number of timesteps in x
        self.y_numsteps =  20 # number of timesteps in y
        # self.human_joints = np.array([5, 8, 11, 28]) # corresponds to [right knee, ankle, foot, big toe]
        self.human_joints = np.array([4, 5, 7, 8, 10, 11, 27, 28]) # corresponds to [right knee, ankle, foot, big toe]
        self.datadir = "/home/fanuc/zheng_work/DLow/raw_ball_tick_data/processed_data"
        self.invalid_seqs = [1, 3, 4, 8, 29, 35, 42, 54, 55, 57, 58, 61, 62, 78, 89, 99,
                                32, 92, 31, 66, 14, 53, 39,26, 48, 21, 5, 74, 97, 85]  # data seqs not in use
        self.all_seqs = ["{:04d}.pkl".format(i) for i in range(1, 101) if i not in self.invalid_seqs]
        import random
        random.Random(4).shuffle(self.all_seqs)
        # self.train_seqs = ["{:04d}.pkl".format(i) for i in range(1, 101) if i < 90 and i not in self.invalid_seqs] # training sequences
        # self.test_seqs = ["{:04d}.pkl".format(i) for i in range(1, 101) if i >= 90 and i not in self.invalid_seqs] # test sequences
        self.train_seqs = self.all_seqs[:-5]
        self.test_seqs = self.all_seqs[-5:]

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            # compute mean
            pass
        else:
            self.mean = mean
            self.std = std

        for seq in self.seq_list:
            # normalize data, (data-mean)/std
            pass
        self.normalized = True

    def __len__(self):
        if self.mode == "train":
            return len(self.train_seqs)
        else:
            return len(self.test_seqs)

    def __getitem__(self, index):
        if self.mode == "train":
            datapath = os.path.join(self.datadir, self.train_seqs[index])
            # print(self.train_seqs[index])
        else:
            datapath = os.path.join(self.datadir, self.test_seqs[index])
        with open(datapath, "rb") as f:
            data = pickle.load(f)
        x_human = data["encode"]["human_traj"][:, self.human_joints, :]
        x_ball_raw = data["encode"]["ball_traj"]
        bh_step_ratio = x_ball_raw.shape[0] / x_human.shape[0]
        x_ball = np.zeros((x_human.shape[0], 3))
        for i in range(x_ball.shape[0]):
            x_ball[i, :] = x_ball_raw[round(i*bh_step_ratio)]
        x_ball = x_ball[:, None, :]
        x_human_ball = np.concatenate([x_human, x_ball], axis=1)

        y_ball_raw = data["decode"]["ball_traj"]
        start_ind = np.random.choice([0,1,2])
        # start_ind = 0
        y_ball = y_ball_raw[start_ind:y_ball_raw.shape[0]:3, :]  # downsample by 3
        if self.if_residual:
            x_human_ball = (x_human_ball[1:, :] - x_human_ball[:-1, :]) / 0.1  # NOTE: to normalize the data
            y_ball = (y_ball[1:, :] - y_ball[:-1, :]) / np.array([[1, 0.1, 0.1]])  # compute the residual

        X = np.zeros((self.x_numsteps, len(self.human_joints)+1, 3)) # human joints + ball, 3d pos
        Y = np.zeros((self.y_numsteps, 3))

        if x_human_ball.shape[0] > self.x_numsteps:
            X = x_human_ball[-self.x_numsteps:, :, :]
        else:
            X[:x_human_ball.shape[0], :, :] = x_human_ball # the rest values are 0

        if y_ball.shape[0] > self.y_numsteps:
            Y = y_ball[:self.y_numsteps, :]
        else:
            Y[:y_ball.shape[0], :] = y_ball
            if self.if_residual:
                pass # the rest values are all 0
            else:
                Y[y_ball.shape[0]:, :] = y_ball[-1:, :]  # the rest values are ball pos of last timestep
        return from_numpy(X), from_numpy(Y), y_ball_raw

    def _teaser(self):
        for seq in self.train_seqs:
            datapath = os.path.join(self.datadir, seq)
            with open(datapath, "rb") as f:
                data = pickle.load(f)
            print("============================")
            print("Human traj len: {}".format(data["encode"]["human_traj"].shape[0]))
            print("Ball encode traj len: {}".format(data["encode"]["ball_traj"].shape[0]))
            print("Ball decode traj len: {}".format(data["decode"]["ball_traj"].shape[0] / 3))


if __name__ == "__main__":
    dataset = DatasetKick(mode="train", if_residual=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        X, Y = data
        # X, Y = dataset.__getitem__(2)
        print(X.shape, Y.shape)




    
