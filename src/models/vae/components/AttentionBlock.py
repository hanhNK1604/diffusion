import torch 
from torch import nn 

class AttentionBlock(nn.Module): 
    def __init__(
        self, 
        in_ch
    ): 
        super(AttentionBlock, self).__init__()
        self.in_ch = in_ch 

        self.norm = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_ch), 
            nn.SiLU(inplace=True)
        )

        self.q = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1) 
        self.k = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1) 
        self.v = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1)

        self.out = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)  
        self.scale = in_ch ** -0.5 

    def forward(self, x): 
        q = self.q(x) 
        k = self.k(x) 
        v = self.v(x) 

        b, c, w, h = x.shape 

        q = q.reshape(b, c, w*h) 
        k = k.reshape(b, c, w*h) 
        v = v.reshape(b, c, w*h) 

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=2) 

        out = torch.einsum('bij,bcj->bci', attn, v)

        out = out.reshape(b, c, w, h) 
        out = self.out(out) 

        return x + out 

# a = torch.rand(size=(1, 256, 32, 32)) 

# net = AttentionBlock(in_ch=256) 
# print(net(a).shape)