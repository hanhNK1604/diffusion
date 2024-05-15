import torch 
from torch import nn 

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_model):
        super(DownsampleBlock, self).__init__()
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.d_model = d_model

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.mp2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Sequential(
            nn.Linear(d_model, out_ch),
            nn.SiLU()
        )
        self.act = nn.SiLU()

    def forward(self, x, t):
        """
        x: (bs, in_ch, h, w)
        t: (bs, d_model)
        """
        t = self.linear(t) # (bs, d_model) => (bs, out_ch)
        t = t.unsqueeze(-1).unsqueeze(-1) # (bs, out_ch) => (bs, out_ch, 1, 1)


        x = self.conv1(x)  # (bs, in_ch, h, w) => (bs, out_ch, h, w)

        x = self.conv2(x)  # (bs, out_ch, h, w) => (bs, out_ch, h, w)
        x = x + t

        x = self.act(x)

        x = self.mp2d(x) # (bs, out_ch, h, w) => (bs, out_ch, h/2, w/2)

        return x

# net = DownsampleBlock(in_ch=64, out_ch=128, d_model = 64)

# x = torch.rand(size=(2, 64, 32, 32))
# t = torch.rand(size=(2, 64))

# print(net(x, t).shape)