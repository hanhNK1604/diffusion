import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 
import torch 
from torch import nn 
from torch.nn import functional as F

from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion
from src.models.components.UNet import UNet

class SuperResolutionDiffusion(UnconditionalDiffusion):
    def __init__(
        self, 
        denoise_net: UNet, 
        time_steps, 
        schedule
    ): 
        super().__init__(denoise_net=denoise_net, time_steps=time_steps, schedule=schedule) 
    
    def forward(self, batch): 
        # hr, lr = batch 
        # noise = torch.randn_like(hr, device=hr.device) 
        # t = torch.randint(low=0, high=self.time_steps, size=(hr.shape[0],), device=hr.device) 
        # self.denoise_net = self.denoise_net.to(hr.device) 

        # x_t = self.forward_process(x=hr, noise=noise, t=t)
        # if torch.randint(low=0, high=2, size=(1,))[0] == 1: 
        #     pred_noise = self.denoise_net.forward(x=x_t, t=t, c=lr) 
        # else: 
        #     pred_noise = self.denoise_net.forward(x=x_t, t=t, c=None) 
            
        # return pred_noise, noise 

        hr, lr = batch 
        noise = torch.randn_like(hr, device=hr.device) 
        t = torch.randint(low=0, high=self.time_steps, size=(hr.shape[0],), device=hr.device) 
        self.denoise_net = self.denoise_net.to(hr.device) 

        up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bilinear") 
        up.required_grad = False

        minus = hr - up 
        minus_t = self.forward_process(x=minus, noise=noise, t=t) 
        pred_noise = self.denoise_net.forward(x=minus_t, t=t, c=lr) 

        return pred_noise, noise


#test 
# hr = torch.randn(size=(4, 3, 256, 256)).to('cuda')
# lr = torch.randn(size=(4, 3, 64, 64)).to('cuda') 

# batch = (hr, lr) 
# denoise_net = UNet(in_ch=3, t_emb_dim=256, base_channel=64, multiplier=[1, 2, 4, 4], use_attention=False, type_condition='sr').to('cuda') 
# srdiff = SuperResolutionDiffusion(denoise_net=denoise_net, time_steps=1000, schedule='cosine') 

# print(srdiff(batch)[0].shape) 
# print(srdiff(batch)[1].shape)

