import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.vae.components.DownsampleBlock import DownsampleBlock # type: ignore
from src.models.vae.components.ResidualBlock import ResidualBlock # type: ignore
from src.models.vae.components.AttentionBlock import AttentionBlock # type: ignore

class Encoder(nn.Module): 
    def __init__(
        self, 
        in_ch: int = 3, 
        z_ch: int = 3,
        base_ch: int = 64,
        multiplier: list = [1, 2, 4],
        double_latent = False 
    ): 
        super(Encoder, self).__init__()

        self.in_ch = in_ch 
        self.z_ch = z_ch 
        self.base_ch = base_ch 
        self.multiplier = multiplier
        self.double_latent = double_latent 

        channels = base_ch

        self.list_ch = [base_ch * i for i in self.multiplier] #[64, 128, 256] 
        self.in_conv = nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=3, padding=1) 

        self.down = [] 

        for i in range(len(self.multiplier)): 

            if i != len(self.multiplier) - 1: 
                downsample = DownsampleBlock(in_ch=channels)  
            else: 
                downsample = nn.Identity() 

            resblock = ResidualBlock(in_ch=channels, out_ch=self.list_ch[i]) 
            self.down.append(downsample)
            self.down.append(resblock) 

            channels = self.list_ch[i] 
        
        self.down = nn.Sequential(*self.down) 

        self.mid = nn.Sequential(
            ResidualBlock(in_ch=channels, out_ch=channels), 
            AttentionBlock(in_ch=channels),
            ResidualBlock(in_ch=channels, out_ch=channels)    
        )

        if self.double_latent: 
            self.out = nn.Sequential(
                nn.GroupNorm(num_channels=channels, num_groups=8), 
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels=channels, out_channels=z_ch*2, kernel_size=3, stride=1, padding=1) 
            )
        
        else: 
            self.out = nn.Sequential(
                nn.GroupNorm(num_channels=channels, num_groups=8), 
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels=channels, out_channels=z_ch, kernel_size=3, stride=1, padding=1) 
            )

    def forward(self, x):
        x = self.in_conv(x) 
        x = self.down(x) 
        x = self.mid(x) 
        x = self.out(x) 

        return x 

# a = torch.rand(size=(4, 3, 256, 256)) 
# net = Encoder(in_ch=3, z_ch=3, double_latent=True) 

# print(net(a).shape)