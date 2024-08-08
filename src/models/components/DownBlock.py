from torch import nn
import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.components.ResBlock import ResBlock
import torch

class DownBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
        )

    def forward(self, x, t):
        """
        params:
          x: batch input: (bs, ch, w, h)
          t: time embedding: (bs, embed_dim=256)
        """
        x = self.down(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        return x + t_emb

#test
# x = torch.rand(size=(32, 64, 128, 128))
# t = torch.rand(size=(32, 256))
# net = DownBlock(inp_ch=64, out_ch=128)
# print(net(x, t).shape)