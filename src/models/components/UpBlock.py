from torch import nn
import torch
import rootutils 

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.ResBlock import ResBlock

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=256, c_emb_dim=256):
        super(UpBlock, self).__init__()

        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential(
            ResBlock(in_ch=in_ch, out_ch=in_ch),
            ResBlock(in_ch=in_ch, out_ch=out_ch), 
            ResBlock(in_ch=out_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch)
        )

        self.c_emb_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(in_features=c_emb_dim, out_features=out_ch) 
        )

    def forward(self, x, skip, t, c=None):

        """
        params:
          x: batch_input: (bs, ch=inp_ch/2, w, h)
          skip: from DownBLock: (bs, ch=inp_ch/2, w*2, h*2)
          t: time_embed: (bs, t_embed_dim=256)
        """
        x = torch.cat([skip, x], dim=1)
        x = self.upsamp(x)
        x = self.up(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        
        if c != None: 
            c_emb = self.c_emb_layers(c)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3]) 
        else: 
            return x + t_emb
        
        return x + t_emb + c_emb
#test
# x = torch.rand(size=(32, 32, 64, 64))
# skip = torch.rand(size=(32, 32, 128, 128))
# t = torch.rand(size=(32, 256))
# c = torch.rand(size=(32, 256))

# net = UpBlock(in_ch=64, out_ch=32)

# print(net(x, skip, t, c).shape)