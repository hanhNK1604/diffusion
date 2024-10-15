from torch import nn
import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.components.ResBlock import ResBlock
import torch

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=256, c_emb_dim=256):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_ch=in_ch, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
        )

        self.c_emb_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(in_features=c_emb_dim, out_features=out_ch)
        )


    def forward(self, x, t, c=None):
        """
        params:
          x: batch input: (bs, ch, w, h)
          t: time embedding: (bs, embed_dim=256)
        """
        x = self.down(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        if c != None: 
            c_emb = self.c_emb_layers(c)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3]) 
        else: 
            return x + t_emb 
        

        return x + t_emb + c_emb

#test
# x = torch.rand(size=(32, 64, 128, 128))
# t = torch.rand(size=(32, 256))
# c = torch.rand(size=(32, 256))
# net = DownBlock(in_ch=64, out_ch=128)
# print(net(x, t, c).shape)
