from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, mid_ch=None, residual=False):
        super(ResBlock, self).__init__()

        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.resnet_conv = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=mid_ch),
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        )
    def forward(self, x):
        """
        params:
          x: batch input: (bs, ch, w, h)
        """

        if self.residual:
            return x + self.resnet_conv(x)
        else:
            return self.resnet_conv(x)

#test
# x = torch.rand(size=(32, 64, 32, 32))
# net = ResBlock(inp_ch=64, out_ch=64, residual=True)
# print(net(x).shape)