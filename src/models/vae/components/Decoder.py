import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.vae.components.UpsampleBlock import UpsampleBlock # type: ignore
from src.models.vae.components.ResidualBlock import ResidualBlock # type: ignore
from src.models.vae.components.AttentionBlock import AttentionBlock # type: ignore

class Decoder(nn.Module): 
    def __init__(
        self, 
        out_ch: int = 3, 
        z_ch: int = 3, 
        base_ch: int = 64,
        multiplier: list = [4, 2, 1] 
    ): 
        super(Decoder, self).__init__() 
        self.out_ch = out_ch 
        self.z_ch = z_ch 
        self.base_ch = base_ch 
        self.multiplier = multiplier

        self.list_ch = [base_ch * i for i in self.multiplier]

        channels = base_ch

        self.in_conv = nn.Conv2d(in_channels=z_ch, out_channels=channels, kernel_size=3, padding=1) 
        self.mid = nn.Sequential(
            ResidualBlock(in_ch=channels, out_ch=channels), 
            AttentionBlock(in_ch=channels), 
            ResidualBlock(in_ch=channels, out_ch=channels)
        )
        
        self.up = [] 
        for i in range(len(self.multiplier)):
            if i != 0: 
                upsample = UpsampleBlock(in_ch=channels) 
            else: 
                upsample = nn.Identity()
            resblock = ResidualBlock(in_ch=channels, out_ch=self.list_ch[i]) 

            self.up.append(upsample)
            self.up.append(resblock)

            channels = self.list_ch[i]

        self.up = nn.Sequential(*self.up) 
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels), 
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x): 
        x = self.in_conv(x) 
        x = self.mid(x) 
        x = self.up(x) 
        x = self.out(x) 

        return x 


# a = torch.rand(size=(4, 3, 64, 64)) 
# net = Decoder(out_ch=3, z_ch=3, base_ch=64) 

# print(net(a).shape)