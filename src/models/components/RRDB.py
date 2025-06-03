import torch 
from torch import nn 
import torch.nn.functional as F 
import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_ch=64, mid_ch=32):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch + mid_ch, out_channels=mid_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_ch + 2 * mid_ch, out_channels=mid_ch, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_ch + 3 * mid_ch, out_channels=mid_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=in_ch + 4 * mid_ch, out_channels=in_ch, kernel_size=3, padding=1)
        self.silu = nn.SiLU(inplace=True)


    def forward(self, x):
        x1 = self.silu(self.conv1(x))
        x2 = self.silu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.silu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.silu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 


class RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, num_block):
        super(RRDB, self).__init__()
        self.main_net = nn.ModuleList([ResidualDenseBlock_5C(in_ch=in_ch, mid_ch=mid_ch) for _ in range(num_block)]) 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1), 
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1), 
            nn.SiLU(inplace=True) 
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        out = x 
        for layer in self.main_net: 
            x = layer(x)
        x = x + out 
        x = F.interpolate(input=x, scale_factor=2, mode="nearest")
        x = self.conv1(x) 
        x = F.interpolate(input=x, scale_factor=2, mode='nearest')
        x = self.conv2(x) 
        x = self.output(x) 
        return x 
        


    
# net = RRDB(in_ch=3, out_ch=64, mid_ch=64, num_block=8).to('cuda') 
# x = torch.rand(size=(4, 3, 64, 64)).to('cuda')
# print(net(x).shape)
# model = net
# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))