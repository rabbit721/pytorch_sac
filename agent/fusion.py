import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Fusion(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_embed_size, text_embed_size, vocab_size):
        super().__init__()

        self.conv1 = nn.Conv2d(state_embed_size, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(text_embed_size, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=2, stride=2)

        #self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.embed = nn.Embedding(vocab_size, state_embed_size)


    def forward(self, inputs):
        x, input_inst = inputs
        # print(x.shape, input_inst.shape)
        x = self.embed(x).permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x_image_rep = F.relu(self.conv2(x))

        input_inst = F.relu(self.conv4(input_inst))
        input_inst_rep = F.relu(self.conv5(input_inst))

        x = x_image_rep*input_inst_rep
        fused = x.reshape(x.size(0), -1)
        return fused
