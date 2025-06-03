import torch 
from torch import nn 
from flow_matching.solver import ODESolver 
from flow_matching.path import CondOTProbPath 
from flow_matching.utils import ModelWrapper 

import rootutils 
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True) 

from src.models.components.UNet import UNet 

class VelocityModel(nn.Module):
    def __init__(self, image_size, channel, net: UNet): 
        super(VelocityModel, self).__init__() 
        self.image_size = image_size 
        self.channel = channel 
        self.net = net
    
    def forward(self, x, t, c=None): 
        velocity = self.net.forward(x=x, t=t, c=c)
        return velocity 

# net = UNet(in_ch=1, type_condition=None, base_channel=64, multiplier=[1, 2, 2, 4], use_attention=True, use_discrete_time=False).to('cuda')
# velocity_model = VelocityModel(image_size=32, channel=1, net=net).to('cuda') 
# x = torch.randn(size=(4, 1, 32, 32)).to("cuda") 
# print(velocity_model(x).shape) 