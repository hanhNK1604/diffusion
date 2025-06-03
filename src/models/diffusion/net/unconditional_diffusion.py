import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)

import torch 
from torch import nn 
import math 
from src.models.components.UNet import UNet 

class UnconditionalDiffusion(nn.Module): 
    def __init__(
        self,
        denoise_net: UNet, 
        time_steps: int, 
        schedule: str, 
    ):
        super(UnconditionalDiffusion, self).__init__() 
        self.denoise_net = denoise_net 
        self.time_steps = time_steps
        self.schedule = schedule

        if schedule == 'linear': 
            self.betas = torch.linspace(start=0.0015, end=0.019, steps=time_steps) 
        
        elif schedule == 'cosine': 
            s = 8e-3
            t = torch.arange(time_steps + 1, dtype=torch.float) / time_steps + s
            alpha_bar = torch.cos(t / (1 + s) * math.pi / 2).pow(2)
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = betas.clamp(max=0.999)
        
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar) 


    def forward_process(self, x, noise, t): 
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(t.device) 
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(t.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None].to(x.device) 
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None].to(x.device) 
        noise = noise.to(x.device) 

        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise 
    
    def forward(self, batch):
        x, label = batch
        t = torch.randint(low=0, high=self.time_steps, size=(x.shape[0],), device=x.device) 
        self.denoise_net = self.denoise_net.to(x.device) 
        noise = torch.randn_like(x, device=x.device) 
        xt = self.forward_process(x, noise, t) 
        pred_noise = self.denoise_net(xt, t) 
        return pred_noise, noise 

    

    
