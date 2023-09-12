import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.BatchNorm2d(channel_dim),
            nn.Conv2d(channel_dim, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(128 * 14 * 14, z_dim)

    def forward(self, x):
        x = self.model(x)
        # x = x.view(x.shape[0], -1)
        # x = self.out(x)
        return x


class Decoder(nn.Module):
    prefix = 'decoder'

    def __init__(self, z_dim, channel_dim):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 256 * 14 * 14)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, channel_dim, 3, 1, 1),
        )
        self.out = nn.Tanh()

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        # x = self.linear(x)
        # x = x.reshape(x.shape[0], 256, 14, 14)
        x = self.model(x)
        x = self.out(x)
        return x


class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim, trans_type='linear'):
        super(Transition, self).__init__()
        if trans_type in ['linear', 'mlp']:
            self.model = TransitionSimple(z_dim, action_dim, trans_type=trans_type)
        elif 'reparam_w' in trans_type:
            self.model = TransitionParam(z_dim, action_dim, hidden_sizes=[64, 64],
                                         orthogonalize_mode=trans_type)
        else:
            raise Exception('Invalid trans_type:', trans_type)

    def forward(self, z, a):
        return self.model(z, a)


class TransitionSimple(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, trans_type='linear'):
        super(TransitionSimple, self).__init__()
        self.trans_type = trans_type
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 64
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x
