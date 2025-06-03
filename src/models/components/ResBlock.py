from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        )

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1) 
        else: 
            self.res_conv = nn.Identity() 
        
    def forward(self, x):
        """
        params:
          x: batch input: (bs, ch, w, h)
        """
        
        out = self.conv(x) 
        res = self.res_conv(x) 

        return out + res 
    



#test
# x = torch.rand(size=(32, 64, 32, 32))
# net = ResBlock(in_ch=64, out_ch=64)
# print(net(x).shape)

# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ResBlock, self).__init__()

        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1), 
#             nn.GroupNorm(num_groups=8, num_channels=out_ch) 
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1), 
#             nn.GroupNorm(num_groups=8, num_channels=out_ch)
#         )


#     def forward(self, x):
#         """
#         params:
#           x: batch input: (bs, ch, w, h)
#         """

#         out1 = self.conv1(x) 
#         out2 = self.conv2(out1) + out1 

#         return out2 