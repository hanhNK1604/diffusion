import torch
from torch import nn 

net = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=4, stride=2, padding=1) 
a = torch.rand(1, 1, 32, 32) 
print(net(a).shape)