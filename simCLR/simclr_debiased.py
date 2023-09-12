# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

import logging
import os
import sys
import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import matplotlib.pyplot as plt

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def get_negative_mask(self):
        negative_mask = torch.ones((self.args.batch_size, 2 * self.args.batch_size), dtype=bool)
        for i in range(self.args.batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + self.args.batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0

        for epoch_counter in range(self.args.epochs):
            for images in tqdm(train_loader):
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # for i in range (16):
                    #     image = images[1, :, i, :, :].cpu().detach()
                    #     fig, ax = plt.subplots()
                    #     ax.imshow(image.permute((1,2,0)))
                    #     plt.show()

                    tuple = (images[:, :, 0:15, :, :], images[:, :, 15:30, :, :])
                    images = torch.cat(tuple, dim=0)
                    features = self.model(images)

                    # neg score
                    neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.args.temperature)
                    mask = self.get_negative_mask().cuda()
                    neg = neg.masked_select(mask).view(2 * self.args.batch_size, -1)

                    # pos score
                    pos = torch.exp(torch.sum(
                        features[0:self.args.batch_size, :] * features[self.args.batch_size:2 * self.args.batch_size,
                                                              :], dim=-1) / self.args.temperature)
                    pos = torch.cat([pos, pos], dim=0)

                    N = self.args.batch_size * 2 - 2
                    Ng = (-self.args.tau_plus * N * pos + neg.sum(dim=-1)) / (1 - self.args.tau_plus)
                    # constrain (optional)
                    Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.args.temperature))

                    # contrastive loss
                    loss = (- torch.log(pos / (pos + Ng))).mean()

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                print(loss)

                # if n_iter % self.args.log_every_n_steps == 0:
                #     top1, top5 = accuracy(logits, labels, topk=(1, 2))
                #     self.writer.add_scalar('loss', loss, global_step=n_iter)
                #     self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                #     self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 2:
                self.scheduler.step()

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

            if epoch_counter % 1 == 0:
                checkpoint_name = 'checkpoint.pth.tar'.format(self.args.epochs)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

    def test(self, train_loader):
        features_all = np.empty([2, 128])  # batch_size=1*2
        checkpoint = torch.load('/home/fanfan/z_physics/data/checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.eval()
        for images in tqdm(train_loader):
            images = images.to(self.args.device)

            with autocast(enabled=self.args.fp16_precision):
                with torch.no_grad():
                    tuple = (images[:, :, 0:15, :, :], images[:, :, 15:30, :, :])
                    images = torch.cat(tuple, dim=0)

                    features = self.model(images)
                    features = features.cpu().detach().numpy()
                    features_all = np.concatenate((features_all, features), axis=0)

        np.save('/home/fanfan/z_physics/data/features_all.npy', features_all)
