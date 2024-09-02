import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 

import torch 
from torch import nn 

from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion 
from src.models.components.UNet import UNet 

class ConditionalDiffusion(UnconditionalDiffusion): 
    def __init__(
        self, 
        denoise_net: UNet, 
        time_steps: int, 
        schedule: str, 
    ): 
        super().__init__(denoise_net=denoise_net, time_steps=time_steps, schedule=schedule) 


    def forward(self, batch): 
        x, label = batch 
        t = torch.randint(low=0, high=self.time_steps, size=(x.shape[0],), device=x.device)
        self.denoise_net = self.denoise_net.to(x.device)

        noise = torch.randn_like(x, device=x.device) 
        xt = self.forward_process(x, noise, t) 

        cond = torch.randint(low=0, high=2, size=(1,)) # 0 is no condition, else yes 
        if cond[0] == 0:
            pred_noise = self.denoise_net(xt, t) 
        else: 
            pred_noise = self.denoise_net(xt, t, label)
        
        return pred_noise, noise 