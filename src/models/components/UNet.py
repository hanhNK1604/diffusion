import torch
from torch import nn

import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.models.components.ResBlock import ResBlock
from src.models.components.DownBlock import DownBlock
from src.models.components.SelfAttentionBlock import SelfAttentionBlock
from src.models.components.UpBlock import UpBlock
from src.models.components.AttentionBlock import AttentionBlock 
from src.models.components.LabelEmbedding import LabelEmbedding  # type: ignore
from src.models.components.ContinuousLabelEmbedding import ContinuousLabelEmbedding
from src.models.components.RRDB import RRDB# type: ignore
from torchsummary import summary
from src.models.rrdb_module import RRDBModule

class UNet(nn.Module):
    def __init__(
        self,
        in_ch,
        t_emb_dim: int = 256,
        base_channel: int = 64,
        multiplier: list = [1, 2, 4], 
        use_attention: bool = True, 
        type_condition: str = None, 
        use_discrete_time: bool = True 
    ):
        super(UNet, self).__init__()

        self.in_ch = in_ch
        self.t_emb_dim = t_emb_dim
        self.base_channel = base_channel
        self.multiplier = multiplier 
        self.use_attention = use_attention
        self.type_condition = type_condition
        self.use_discrete_time = use_discrete_time

        self.list_channel_down = [base_channel * i for i in self.multiplier]                #[64, 128, 256, 512]
        self.list_channel_up = [base_channel * i * 2 for i in reversed(self.multiplier)]    #[1024, 512, 256, 128]

        self.time_continous_embed = None 
        if not self.use_discrete_time: 
            self.time_continous_embed = nn.Sequential(
                nn.Linear(in_features=1, out_features=self.t_emb_dim),
                nn.SiLU(),
                nn.Linear(in_features=self.t_emb_dim, out_features=self.t_emb_dim), 
                nn.SiLU() 
            )  
        
        if type_condition == 'label': 
            self.embedder = LabelEmbedding(num_embeds=10, emb_dim=t_emb_dim) 
    
        if type_condition == "sr": 
            self.embedder = RRDBModule.load_from_checkpoint('/mnt/apple/k66/hanh/diffusion/logs/train/runs/rrdb/rrdb/lrkx9qsh/checkpoints/epoch=99-step=700000.ckpt')
            self.embedder.eval().freeze()
            for p in self.embedder.parameters(): 
                p.requires_grad = False
        
        if type_condition == "continuous_label": 
            self.embedder = ContinuousLabelEmbedding(emb_dim=t_emb_dim)

        self.inp = nn.Sequential(
            ResBlock(in_ch=in_ch*2 if self.type_condition=='sr' else in_ch, out_ch=base_channel), 
            nn.Mish() if self.type_condition == 'sr' else nn.SiLU()
        )

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        channel = base_channel
        for i in range(len(self.list_channel_down)): 
            down = DownBlock(in_ch=channel, out_ch=self.list_channel_down[i], t_emb_dim=self.t_emb_dim)
            attn = AttentionBlock(channels=self.list_channel_down[i]) if self.use_attention else nn.Identity()
            self.down.append(down) 
            self.down.append(attn) 

            channel = self.list_channel_down[i] 

        self.latent = nn.Sequential(
            ResBlock(in_ch=channel, out_ch=channel), 
            AttentionBlock(channels=channel) if self.use_attention else nn.Identity(), 
            ResBlock(in_ch=channel, out_ch=channel)
        )

        for i in range(len(self.list_channel_up)):
            if i == len(self.list_channel_up) - 1: 
                up = UpBlock(in_ch=self.list_channel_up[i], out_ch=self.base_channel, t_emb_dim=self.t_emb_dim)
                attn = AttentionBlock(channels=self.base_channel) if self.use_attention else nn.Identity()
            else:
                up = UpBlock(in_ch=self.list_channel_up[i], out_ch=self.list_channel_down[len(self.list_channel_up) - 2 - i], t_emb_dim=self.t_emb_dim) 
                attn = AttentionBlock(channels=self.list_channel_down[len(self.list_channel_up) - 2 - i]) if self.use_attention else nn.Identity()
            
            self.up.append(up) 
            self.up.append(attn) 
        
        self.out = nn.Sequential( 
            nn.GroupNorm(num_groups=8, num_channels=self.base_channel),
            nn.SiLU(inplace=True),  
            nn.Conv2d(in_channels=self.base_channel, out_channels=in_ch, kernel_size=3, padding=1) 
        )

    def position_embeddings(self, t, channels):
        i = 1 / (10000 ** (torch.arange(start=0, end=channels, step=2) / channels)).to(t.device)
        pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t, c=None):
        t = t.view(-1, 1).float().to(x.device)
        if self.use_discrete_time: 
            t = self.position_embeddings(t, self.t_emb_dim)
        else: 
            t = self.time_continous_embed(t) 
        
        if c is not None and self.type_condition == 'label': 
            c = self.embedder.forward(c) 
            x = self.inp(x)

        elif c is not None and self.type_condition == "continuous_label": 
            c = self.embedder.forward(c) 
            x = self.inp(x) 
        
        elif c is not None and self.type_condition == 'sr': 
            condition_embed = self.embedder.forward((None, c))
            x = torch.cat((x, condition_embed), dim=1)
            x = self.inp(x) 
            c = None
        
        elif c is None and self.type_condition == 'sr': 
            x = torch.cat((x, x), dim=1)
            x = self.inp(x)

        else:
            x = self.inp(x) 

        output_down = [] 

        for i in range(0, len(self.down), 2): 
            x = self.down[i](x, t, c) 
            x = self.down[i + 1](x) 
            output_down.append(x) 

        x = self.latent(x)

        for i in range(0, len(self.up), 2):
            x = self.up[i](x, output_down[-1], t, c) 
            x = self.up[i + 1](x) 
            output_down.pop()   

        x = self.out(x) 

        return x 

    
#test 

# net = UNet(in_ch=1, type_condition="continuous_label", base_channel=32, multiplier=[1, 2, 4], use_attention=True, use_discrete_time=False).to('cuda')

# x = torch.rand(size=(4, 1, 128, 128)).to('cuda')
# t = torch.rand(size=(4,)).to('cuda')
# c = torch.rand(size=(4,)).to('cuda')
# print(net(x, t, c).shape)

# model = net
# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))
