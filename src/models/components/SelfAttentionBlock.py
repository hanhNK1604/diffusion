from torch import nn
import torch

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()

        self.attn_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x):
        """
        params:
          x: batch input : (bs, ch, w, h)
        """
        b, c, h, w = x.shape

        inp_attn = x.reshape(b, c, h*w)
        inp_attn = self.attn_norm(inp_attn)

        inp_attn = inp_attn.transpose(1, 2)
        out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
        out_attn = out_attn.transpose(1, 2).reshape(b, c, h, w)
        return x + out_attn

#test
# x = torch.rand(size=(32, 32, 32, 32))
# net = SelfAttentionBlock(channels=32)
# print(net(x).shape)