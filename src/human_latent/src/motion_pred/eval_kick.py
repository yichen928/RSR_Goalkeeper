import numpy as np
import argparse
import os
import sys
import pickle
import csv
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from dlow_utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_kick import DatasetKick
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

cm = plt.get_cmap('gist_rainbow')
num_colors = 20
COLORS = [cm(i/num_colors) for i in range(num_colors)]

def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(X, algo, sample_num, num_seeds=1):
    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        X = X.repeat_interleave(nk, dim=1)
        Y = models['vae'].decode(X, Z_g)
    elif algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        # import pdb; pdb.set_trace()
        Y = models[algo].sample_prior(X)

    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y

def plot_prediction(gt_Y, pred_Ys, id, if_residual=False):
    gt_Y = gt_Y.squeeze()
    fig, axes = plt.subplots(1,2)
    axes[0].plot(gt_Y[:, 0], gt_Y[:, 1])
    axes[1].plot(gt_Y[:, 0], gt_Y[:, 2])
    for i in range(pred_Ys.shape[0]):
        pred_Y = pred_Ys[i, :, :]
        if if_residual:
            # import pdb; pdb.set_trace()
            pred_Y = np.cumsum(pred_Y*np.array([[1, 0.1, 0.1]]), axis=0) + gt_Y[0:1, :].cpu().numpy()
        axes[0].plot(pred_Y[:, 0], pred_Y[:, 1], '--', color=COLORS[i])
        axes[1].plot(pred_Y[:, 0], pred_Y[:, 2], '--', color=COLORS[i])
    savedir = "/home/fanuc/zheng_work/DLow/results/human_kick/vis"
    plt.savefig(os.path.join(savedir, "out_{:03d}.png".format(id)))

def plot_prediction_train(gt_Y, pred_Ys, id, if_residual=False):
    gt_Y = gt_Y.squeeze()
    fig, axes = plt.subplots(1,3)
    axes[0].plot(np.arange(len(gt_Y[:, 0])), gt_Y[:, 0])
    axes[1].plot(np.arange(len(gt_Y[:, 0])), gt_Y[:, 1])
    axes[2].plot(np.arange(len(gt_Y[:, 0])), gt_Y[:, 2])
    for i in range(pred_Ys.shape[0]):
        pred_Y = pred_Ys[i, :, :]
        if if_residual:
            # import pdb; pdb.set_trace()
            pred_Y = np.cumsum(pred_Y*np.array([[1, 0.1, 0.1]]), axis=0) + gt_Y[0:1, :].cpu().numpy()
        axes[0].plot(np.arange(len(pred_Y[:, 0])), pred_Y[:, 0], '--', color=COLORS[i])
        axes[1].plot(np.arange(len(pred_Y[:, 0])), pred_Y[:, 1], '--', color=COLORS[i])
        axes[2].plot(np.arange(len(pred_Y[:, 0])), pred_Y[:, 2], '--', color=COLORS[i])
    savedir = "/home/fanuc/zheng_work/DLow/results/human_kick/vis"
    plt.savefig(os.path.join(savedir, "train_out_{:03d}.png".format(id)))


if __name__ == '__main__':

    # all_algos = ['dlow', 'vae']
    all_algos = ['vae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=-1)
    parser.add_argument('--use_residual', action='store_true', default=False)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetKick
    # dataset = dataset_cls('test', args.use_residual)
    dataset = dataset_cls('train', args.use_residual)
    if cfg.normalize_data:
        dataset.normalize_data()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    """models"""
    model_generator = {
        'vae': get_kick_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    for i, data in enumerate(dataloader):
        X, Y, raw_y = data
        X = X.reshape(X.shape[0], X.shape[1], -1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
        X = tensor(X, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        # Y = tensor(Y, device=device, dtype=dtype).permute(1, 0, 2).contiguous()

        pred_Y = get_prediction(X, algo, sample_num=1).squeeze(axis=0) # axis 0 is number of seeds
        # import pdb; pdb.set_trace()
        gt_Y = raw_y
        print("======================")
        print("plotting for {}-th data point".format(i))
        plot_prediction_train(gt_Y, pred_Y, i, dataset.if_residual)



