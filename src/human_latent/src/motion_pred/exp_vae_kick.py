import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from dlow_utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_kick import DatasetKick
from models.motion_pred import *


def loss_function(X, Y_r, Y, mu, logvar):
    MSE = (Y_r - Y).pow(2).sum() / Y.shape[1]
    # MSE_v = (X[-1] - Y_r[0]).pow(2).sum() / Y.shape[1] # probably no need?
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]
    # loss_r = MSE + cfg.lambda_v * MSE_v + cfg.beta * KLD
    loss_r = MSE + cfg.beta * KLD
    # return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])
    return loss_r, np.array([loss_r.item(), MSE.item(), KLD.item()])



def train(epoch, dataloader):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'KLD']
    # generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    # for traj_np in generator:
    for i, data in enumerate(dataloader):
        X, Y, _ = data
        X = X.reshape(X.shape[0], X.shape[1], -1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
        X = tensor(X, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        Y = tensor(Y, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        # traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        # X = traj[:t_his]
        # Y = traj[t_his:]
        # import pdb; pdb.set_trace()
        Y_r, mu, logvar = model(X, Y)
        loss, losses = loss_function(X, Y_r, Y, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    # for name, loss in zip(loss_names, train_losses):
    #     tb_logger.add_scalar('vae_' + name, loss, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    # t_his = cfg.t_his
    # t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetKick
    dataset = dataset_cls('train', if_residual=args.use_residual)
    if cfg.normalize_data:
        dataset.normalize_data()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    """model"""
    # model = get_vae_model(cfg, dataset.traj_dim)
    model = get_kick_vae_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i, dataloader)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))



