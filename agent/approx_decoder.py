import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils

class Approximator(nn.Module):
    def __init__(self, fusion_dim, hidden_dim, hidden_depth, latent_dim):
        super().__init__()

        self.linear_module = utils.mlp(fusion_dim * 2, hidden_dim, latent_dim,
                                       hidden_depth)

    def forward(self, state_pair):
        (st, stp1) = state_pair
        # print(torch.cat((st, stp1), dim=1).shape)
        return self.linear_module(torch.cat((st, stp1), dim=1))


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, hidden_depth, action_dim):
        super().__init__()

        self.linear_module = utils.mlp(latent_dim, hidden_dim, action_dim,
                                       hidden_depth)

    def forward(self, cont_vec):
        # print(cont_vec)
        conv_vec_ln = self.linear_module(cont_vec)
        # print(conv_vec_ln)
        prob = F.softmax(conv_vec_ln, dim=1)
        if prob.shape[0] == 1:
            print("#### action prob: {} ####".format(prob))
        return prob
