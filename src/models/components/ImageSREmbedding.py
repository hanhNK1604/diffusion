import torch 
from torch import nn 

class ImageSREmbedding(nn.Module):
    def __init__(
        self, 
        in_ch
    ): 
        super(ImageSREmbedding, self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1), 
        )
    
    def forward(self, x): 
        return x + self.conv(x) 