import torch
from torch import nn

import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.components.ResBlock import ResBlock
from src.models.components.DownBlock import DownBlock
from src.models.components.SelfAttentionBlock import SelfAttentionBlock
from src.models.components.UpBlock import UpBlock
from src.models.components.AttentionBlock import AttentionBlock 
from src.models.components.LabelEmbedding import LabelEmbedding  # type: ignore
from src.models.components.ImageSREmbedding import ImageSREmbedding  # type: ignore


class UNetNoAttn(nn.Module):
    def __init__(
        self,
        in_ch,
        t_emb_dim: int = 256,
        type_condition: str = None  
    ):
        super(UNetNoAttn, self).__init__()

        self.in_ch = in_ch
        self.t_emb_dim = t_emb_dim
        self.type_condition = type_condition
        
        if type_condition == 'label': 
            self.embedder = LabelEmbedding(num_embeds=10, emb_dim=t_emb_dim)
        
        if type_condition == 'image_sr': 
            self.embedder = ImageSREmbedding(in_ch=3)


        self.inp = ResBlock(in_ch=in_ch, out_ch=64)

        self.down1 = DownBlock(in_ch=64, out_ch=128)
        # self.sa1 = AttentionBlock(channels=128)
        self.down2 = DownBlock(in_ch=128, out_ch=256)
        # self.sa2 = AttentionBlock(channels=256)
        self.down3 = DownBlock(in_ch=256, out_ch=256)
        # self.sa3 = AttentionBlock(channels=256)

        self.lat1 = ResBlock(in_ch=256, out_ch=256)
        # self.sa_la = AttentionBlock(channels=256)
        self.lat2 = ResBlock(in_ch=256, out_ch=256)

        self.up1 = UpBlock(in_ch=512, out_ch=128)
        # self.sa4 = AttentionBlock(channels=128)
        self.up2 = UpBlock(in_ch=256, out_ch=64)
        # self.sa5 = AttentionBlock(channels=64)
        self.up3 = UpBlock(in_ch=128, out_ch=64)
        # self.sa6 = AttentionBlock(channels=64)


        self.out = nn.Sequential( 
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SiLU(inplace=True),  
            nn.Conv2d(in_channels=64, out_channels=in_ch, kernel_size=3, padding=1) 
        )

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2) / channels)).to(t.device)
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t, c=None):
        t = t.unsqueeze(1).float().to(x.device)
        t = self.position_embeddings(t, self.t_emb_dim)

        if c != None and self.type_condition == 'image_sr': 
            x = x + self.embedder(c) 
            c = None 
        
        if c != None and self.type_condition == 'label': 
            c = self.embedder(c) 

        x1 = self.inp(x)

        x2 = self.down1(x1, t, c)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t, c)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t, c)
        # x4 = self.sa3(x4)

        x4 = self.lat1(x4)
        # x4 = self.sa_la(x4)
        x4 = self.lat2(x4)

        x = self.up1(x4, x3, t, c)
        # x = self.sa4(x)
        x = self.up2(x, x2, t, c)
        # x = self.sa5(x)
        x = self.up3(x, x1, t, c)
        # x = self.sa6(x)

        output = self.out(x)

        return output 

#test 

# net = UNetNoAttn(in_ch=3, type_condition='image_sr').to('cuda')

# x = torch.rand(size=(10, 3, 64, 64)).to('cuda')
# t = torch.rand(size=(10,)).to('cuda') 
# # c = torch.randint(low=0, high=10, size=(10,))
# c = torch.rand(10, 3, 64, 64).to('cuda') 

# print(net(x, t, c).shape)