from torch import nn
import torch
import rootutils 

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.ResBlock import ResBlock

class UpBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256, c_emb_dim=256):
        super(UpBlock, self).__init__()

        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Sequential(
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch, mid_ch=inp_ch//2)
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

        x = self.upsamp(x)
        x = torch.cat([skip, x], dim=1)
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