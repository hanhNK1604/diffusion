import torch 
from torch import nn 

class UpsampleBlock(nn.Module):
  def __init__(self, in_ch, out_ch, d_model):
    super(UpsampleBlock, self).__init__()
    self.in_channel = in_ch
    self.out_channel = out_ch
    self.d_model = d_model

    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

    self.linear = nn.Sequential(
        nn.Linear(d_model, out_ch),
        nn.SiLU()
    )

    self.act = nn.SiLU()

  def forward(self, x, op_x, t):
    t = self.linear(t).unsqueeze(-1).unsqueeze(-1)
    x = self.conv1(x)
    x = torch.cat((op_x, x), dim=1)
    x = self.conv2(x)
    x = x + t
    return x

# net = UpsampleBlock(in_ch=1024, out_ch=512, d_model=64)
# x = torch.rand(size=(2, 1024, 2, 2))
# op_x = torch.rand(2, 512, 4, 4)
# t = torch.rand(2, 64)

# print(net(x, op_x, t).shape)


