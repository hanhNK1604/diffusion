import torch 
from torch import nn 

class UpsampleBlock(nn.Module): 
    def __init__(
        self, 
        in_ch: int 
    ): 
        super(UpsampleBlock, self).__init__() 
        self.in_ch = in_ch 
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1) 

    def forward(self, x): 
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x) 
    
# a = torch.rand(size=(1, 3, 32, 32)) 
# net = UpsampleBlock(in_ch=3) 
# print(net(a).shape)