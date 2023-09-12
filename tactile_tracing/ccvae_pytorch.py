'''ccvae
'''

import sys

sys.dont_write_bytecode = True
sys.path.append('/home/fanfan/z_123')

import argparse
import json
import os
from os.path import join, exists
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import models as cm
from termcolor import colored
from torch.autograd import Variable

batch_size = 8
img_rows, img_cols = 224, 224

x_all = np.load('/home/fanfan/Desktop/point2/Dataset_rgb_self.npy') / 255.0
x_all_pos = np.load('/home/fanfan/Desktop/point2/Dataset_rgb_pos.npy') / 255.0
y_all = np.load('/home/fanfan/Desktop/point2/action.npy')
num = x_all.shape[0]
n_batch = num / batch_size
x_all = x_all.reshape(num, 3, img_rows, img_cols)
x_all_pos = x_all_pos.reshape(num, 3, img_rows, img_cols)
y_all = y_all.reshape(num, 3)
train_x = torch.from_numpy(x_all)
train_x_pos = torch.from_numpy(x_all_pos)
train_y = torch.from_numpy(y_all)
print(train_x.shape)
print(train_y.shape)


def compute_cpc_loss(obs, obs_pos, encoder, decoder, trans, actions, device):
    # bs = obs.shape[0]

    # z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    # z_next = trans(z, actions)
    # obs_rec = decoder(z_next)
    #
    # neg_dot_products = torch.mm(z_next, z.t())  # b x b
    # neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2 * neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    #
    # pos_dot_products = (z_pos * z_next).sum(dim=1)  # b
    # pos_dists = -((z_pos ** 2).sum(1) - 2 * pos_dot_products + (z_next ** 2).sum(1))
    # pos_dists = pos_dists.unsqueeze(1)  # b x 1
    #
    # dists = torch.cat((neg_dists, pos_dists), dim=1)  # b x b+1
    # dists = F.log_softmax(dists, dim=1)  # b x b+1
    # loss_cpc = -dists[:, -1].mean()  # Get last column with is the true pos sample
    # L = torch.nn.MSELoss()
    # loss_rec = L(obs_pos, obs_rec)
    # loss = 0*loss_cpc + loss_rec

    ############################################# reconstruct autoencoder
    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    obs_rec = decoder(z)
    L = torch.nn.MSELoss()
    loss_rec = L(obs, obs_rec)
    loss_cpc = loss_rec
    loss = loss_rec

    return loss_cpc, loss_rec, loss


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (3, 224, 224)
    action_dim = 3

    device = torch.device('cuda')

    encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
    decoder = cm.Decoder(args.z_dim, obs_dim[0]).to(device)
    trans = cm.Transition(args.z_dim, action_dim, trans_type=args.trans_type).to(device)
    parameters = list(encoder.parameters()) + list(trans.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    encoder.train()
    trans.train()
    decoder.train()

    for epoch in range(args.epochs):
        parameters = list(encoder.parameters()) + list(trans.parameters()) + list(decoder.parameters())

        for i in range(int(n_batch)):
            # Local batches and labels
            obs, obs_pos, actions = Variable(train_x[i * batch_size:(i + 1) * batch_size, :, :, :]), \
                                    Variable(train_x_pos[i * batch_size:(i + 1) * batch_size, :, :, :]), \
                                    Variable(train_y[i * batch_size:(i + 1) * batch_size, :])
            obs = obs.to(device)
            obs_pos = obs_pos.to(device)
            actions = actions.to(device)

            obs = obs.float()
            obs_pos = obs_pos.float()
            actions = actions.float()

            loss_cpc, loss_rec, loss = compute_cpc_loss(obs, obs_pos, encoder, decoder, trans, actions, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 20)
            optimizer.step()

        print(colored('epoch {} loss_cpc {} loss_rec {} loss {}'.format(epoch, loss_cpc, loss_rec, loss), color='blue',
                      attrs=['bold']))

        if epoch % args.log_interval == 0:
            checkpoint = {
                'encoder': encoder.state_dict(),
                'trans': trans.state_dict(),
                'optimizer': optimizer.state_dict(),
                'decoder': decoder.state_dict(),
            }
            torch.save(checkpoint, join(folder_name, 'checkpoint'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_type', type=str, default='mlp',
                        help='linear | mlp | reparam_w | reparam_w_ortho_gs | reparam_w_ortho_cont | reparam_w_tanh (default: linear)')
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5 * 1e-4, help='default 0')
    parser.add_argument('--epochs', type=int, default=100, help='default: 50')
    parser.add_argument('--log_interval', type=int, default=10, help='default: 1')
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8, help='default 128')
    parser.add_argument('--z_dim', type=int, default=256, help='dimension of the latents')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='', help='folder name results are stored into')
    args = parser.parse_args()

    main()
