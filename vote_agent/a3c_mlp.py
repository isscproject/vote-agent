import torch.nn as nn
import torch

class A3C_MLP_NET(nn.Module):
    def __init__(self):
        super(A3C_MLP_NET, self).__init__()
        self.fc1 = nn.Linear(39, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(256, 64)
        self.fc3_2 = nn.Linear(64, 6)
        self.fc4_1 = nn.Linear(256, 64)
        self.fc4_2 = nn.Linear(64, 1)

    def forward(self, din):
        # print(type(din),din)
        din = din.view(-1, 39)
        dout = torch.tanh(self.fc1(din))
        dout = torch.tanh(self.fc2(dout))
        dactor = torch.tanh(self.fc3_1(dout))
        dactor = self.fc3_2(dactor)
        dcritic = torch.tanh(self.fc4_1(dout))
        dcritic = self.fc4_2(dcritic)
        return dactor, dcritic
