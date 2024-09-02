import torch 
from torch import nn 

class DownsampleBlock(nn.Module): 
    def __init__(
        self, 
        in_ch
    ): 
        super(DownsampleBlock, self).__init__() 
        self.in_ch = in_ch 

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, stride=2) 
    
    def forward(self, x): 
        return self.conv(x) 


# a = torch.rand(size=(1, 3, 32, 32)) 
# net = DownsampleBlock(in_ch=3) 
# print(net(a).shape)