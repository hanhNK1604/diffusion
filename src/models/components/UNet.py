import torch
from torch import nn

import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.components.ResBlock import ResBlock
from src.models.components.DownBlock import DownBlock
from src.models.components.SelfAttentionBlock import SelfAttentionBlock
from src.models.components.UpBlock import UpBlock


class UNet(nn.Module):
    def __init__(
        self,
        inp_ch,
        t_emb_dim,
    ):
        super(UNet, self).__init__()

        self.inp_ch = inp_ch
        self.t_emb_dim = t_emb_dim

        self.inp = ResBlock(inp_ch=inp_ch, out_ch=64)

        self.down1 = DownBlock(inp_ch=64, out_ch=128)
        self.sa1 = SelfAttentionBlock(channels=128)
        self.down2 = DownBlock(inp_ch=128, out_ch=256)
        self.sa2 = SelfAttentionBlock(channels=256)
        self.down3 = DownBlock(inp_ch=256, out_ch=256)
        self.sa3 = SelfAttentionBlock(channels=256)

        self.lat1 = ResBlock(inp_ch=256, out_ch=512)
        self.sa_la1 = SelfAttentionBlock(channels=512)
        self.lat2 = ResBlock(inp_ch=512, out_ch=512)
        self.sa_la2 = SelfAttentionBlock(channels=512)
        self.lat3 = ResBlock(inp_ch=512, out_ch=256)

        self.up1 = UpBlock(inp_ch=512, out_ch=128)
        self.sa4 = SelfAttentionBlock(channels=128)
        self.up2 = UpBlock(inp_ch=256, out_ch=64)
        self.sa5 = SelfAttentionBlock(channels=64)
        self.up3 = UpBlock(inp_ch=128, out_ch=64)
        self.sa6 = SelfAttentionBlock(channels=64)
        # TODO: how to change model easily

        self.out = nn.Conv2d(in_channels=64, out_channels=inp_ch, kernel_size=1)

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2) / channels)).to(t.device)
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t):
        t = t.unsqueeze(1).float().to(x.device)
        t = self.position_embeddings(t, self.t_emb_dim)

        x1 = self.inp(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.lat1(x4)
        x4 = self.sa_la1(x4)
        x4 = self.lat2(x4)
        x4 = self.sa_la2(x4)
        x4 = self.lat3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.out(x)
        return output

