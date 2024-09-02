import torch 
from torch import nn

class ResidualBlock(nn.Module): 
    def __init__(
        self, 
        in_ch, 
        out_ch,
    ):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch 
        self.out_ch = out_ch 

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_ch), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1) 
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_ch), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1) 
        )

        if in_ch == out_ch: 
            self.skip_connection = nn.Identity() 
        else: 
            self.skip_connection = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1) 


    def forward(self, x): 
        out = self.conv1(x) 
        out = self.conv2(out) 

        return self.skip_connection(x) + out 
        

# net = ResidualBlock(in_ch=32, out_ch=64) 
# a = torch.rand(1, 32, 32, 32) 
# print(net(a).shape)