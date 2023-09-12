#!/usr/bin/python

import torch
import torchvision
import sys
import cv2
import random
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from prettytable import PrettyTable
from collections import namedtuple
from collections import OrderedDict
from torch.autograd import Variable
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

out_resolution = 224

x_all = np.load('./data_2/x.npy') / 255
y_all = np.load('./data_2/Y_heatmap.npy') * 100

x = []
y = []
for i in range(x_all.shape[0]):
    if np.max(y_all[i, :, :]) > 0:
        x.append(x_all[i, :, :, :])
        y.append(y_all[i, :, :])
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

# copy y three times

num = y.shape[0]
x_shu, y_shu = shuffle(x, y)
x_train = x_shu[0:num, :]
y_train = y_shu[0:num, :]
x_test = x_shu[0:num, :]
y_test = y_shu[0:num, :]

train_x = x
train_x = torch.from_numpy(train_x)
train_y = y
train_y = torch.from_numpy(train_y)

val_x = x
val_x = torch.from_numpy(val_x)
val_y = y
val_y = torch.from_numpy(val_y)

learning_rate = 0.0001
batch_size = 4
n_batch = 45 / batch_size
num_epochs = 1000
margin = 0.2


class Grasping_Module_multidiscrete(nn.Module):
    def __init__(self, output_activation='Sigmoid'):
        super(Grasping_Module_multidiscrete, self).__init__()
        self.pushleft_color_trunk = torchvision.models.densenet121(pretrained=True)
        self.pushleft_depth_trunk = torchvision.models.densenet121(pretrained=True)
        self.grasp_color_trunk = torchvision.models.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet121(pretrained=True)
        self.pushright_color_trunk = torchvision.models.densenet121(pretrained=True)
        self.pushright_depth_trunk = torchvision.models.densenet121(pretrained=True)

        # Construct network branches for pushing and grasping
        self.leftpushnet = nn.Sequential(OrderedDict([
            ('pushleft-norm0', nn.BatchNorm2d(1024)),
            ('pushleft-relu0', nn.ReLU(inplace=True)),
            ('pushleft-conv0', nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False)),
            ('pushleft-norm1', nn.BatchNorm2d(64)),
            ('pushleft-relu1', nn.ReLU(inplace=True)),
            ('pushleft-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('pushleft-upsample2', nn.Upsample(size=(out_resolution, out_resolution), mode='bilinear')),
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1024)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('grasp-upsample2', nn.Upsample(size=(out_resolution, out_resolution), mode='bilinear')),
        ]))
        self.rightpushnet = nn.Sequential(OrderedDict([
            ('pushright-norm0', nn.BatchNorm2d(1024)),
            ('pushright-relu0', nn.ReLU(inplace=True)),
            ('pushright-conv0', nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False)),
            ('pushright-norm1', nn.BatchNorm2d(64)),
            ('pushright-relu1', nn.ReLU(inplace=True)),
            ('pushright-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('pushright-upsample2', nn.Upsample(size=(out_resolution, out_resolution), mode='bilinear')),
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'pushright-' in m[0] or 'grasp-' in m[0] or 'pushleft-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # camera number
        self.num_rotations = 1

    def forward(self, x, verbose=0):
        for rotate_idx in range(self.num_rotations):
            rotate_color = x[:, 3 * rotate_idx: 3 * rotate_idx + 3, :, :]
            # rotate_depth = x[:, 3 + 3 * rotate_idx: 3 + 3 * rotate_idx + 3, :, :]

            color_tensor = torch.empty(rotate_color.shape[0], 3, 224, 224, device="cuda:0")
            # depth_tensor = torch.empty(rotate_depth.shape[0], 3, 224, 224, device="cuda:0")
            for i in range(rotate_color.shape[0]):
                color_tensor[i, :, :, :] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    rotate_color[i, :, :, :])
                # depth_tensor[i, :, :, :] = transforms.Normalize(mean=[0.01, 0.01, 0.01], std=[0.03, 0.03, 0.03])(
                #     rotate_depth[i, :, :, :])

            # Compute intermediate features
            interm_pushleft_color_feat = self.pushleft_color_trunk.features(color_tensor)
            # interm_pushleft_depth_feat = self.pushleft_depth_trunk.features(rotate_depth)
            # interm_pushleft_feat = torch.cat((interm_pushleft_color_feat, interm_pushleft_depth_feat), dim=1)
            interm_pushleft_feat = interm_pushleft_color_feat

            interm_grasp_color_feat = self.grasp_color_trunk.features(color_tensor)
            # interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            # interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            interm_grasp_feat = interm_grasp_color_feat

            interm_pushright_color_feat = self.pushright_color_trunk.features(color_tensor)
            # interm_pushright_depth_feat = self.pushright_depth_trunk.features(rotate_depth)
            # interm_pushright_feat = torch.cat((interm_pushright_color_feat, interm_pushright_depth_feat), dim=1)
            interm_pushright_feat = interm_pushright_color_feat

            if rotate_idx == 0:
                self.output_prob = torch.cat(
                    (self.leftpushnet(interm_pushleft_feat),
                     self.graspnet(interm_grasp_feat),
                     self.rightpushnet(interm_pushright_feat)), dim=2)
            else:
                self.tmp = torch.cat(
                    (self.leftpushnet(interm_pushleft_feat),
                     self.graspnet(interm_grasp_feat),
                     self.rightpushnet(interm_pushright_feat)), dim=2)
                self.output_prob = torch.cat((self.output_prob, self.tmp), dim=2)

        return self.output_prob


policy_net = Grasping_Module_multidiscrete().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=0.00002)

########################################################################
# # # Train the model
for epoch in range(num_epochs):
    for i in range(int(n_batch)):
        # Local batches and labels
        x_train, y_train = Variable(train_x[i * batch_size:(i + 1) * batch_size, :, :, :]), \
                           Variable(train_y[i * batch_size:(i + 1) * batch_size, :])
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # x_val, y_val = Variable(val_x), Variable(val_y)
        # x_val = x_val.to(device)
        # y_val = y_val.to(device)

        # Q pass
        output_train = policy_net(x_train)
        output_train_q = output_train.reshape(batch_size, 3 * out_resolution * out_resolution)
        y_flat = y_train.view(-1)
        indices = ((y_flat == 1).nonzero(as_tuple=True))
        q_pred = output_train_q.view(-1)[indices]
        q_expected = y_flat[indices]
        q_loss = F.smooth_l1_loss(q_pred, q_expected)

        # # supervised loss
        # num_actions = output_train.size(1) * output_train.size(2) * output_train.size(3)
        # margins = (torch.ones(batch_size, num_actions)) * margin
        # margins[0, indices] = 0
        # batch_margins = margins
        # output_train_s = output_train.view(batch_size, -1) + Variable(batch_margins).type(dtype)
        # supervised_loss = (output_train_s.max(1)[0].unsqueeze(1) - q_pred).pow(2).sum()
        # print('q_pred', q_pred)
        # print('ss', output_train.view(batch_size, -1).max(1)[0].unsqueeze(1))

        # supervised loss
        supervised_loss = (output_train_q.view(-1) - y_train.view(-1)).pow(2).sum()

        loss = q_loss + 0.3 * supervised_loss # need tune

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('q_pred', q_pred)
    print(colored('epoch {} loss_train {} q_loss {} supervised_loss {}'.format(epoch, loss, q_loss, supervised_loss),
                  color='blue', attrs=['bold']))

PATH = './model_q/policy_net_model.pth'
torch.save(policy_net, PATH)
print(colored('saved', color='green', attrs=['bold']))
# # PATH = './model_q/policy_net_model.pth'
# # policy_net = torch.load(PATH)
#
PATH = './model_q/policy_net_state.pth'
torch.save(policy_net.state_dict(), PATH)
sys.exit(0)

########################################################################
# # # # later...
PATH = './model_q/policy_net_state.pth'
policy_net.load_state_dict(torch.load(PATH))
policy_net.eval()
x_val = Variable(train_x[0:1, :, :, :])
x_val = x_val.to(device)
outputs = policy_net(x_val)

# test_in = train_x[5:6, 0:3, :, :].reshape(224, 224, 3)
outputs = outputs.reshape(18, out_resolution, out_resolution)
# outputs = train_y[0, :].reshape(18, out_resolution, out_resolution)
test_out = outputs[15, :, :].detach().cpu().numpy()

fig, ax = plt.subplots()
ax.imshow(test_out)
plt.show()
sys.exit(0)
