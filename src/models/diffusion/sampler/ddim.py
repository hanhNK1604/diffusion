import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

import torch 
from tqdm import tqdm 

from src.models.diffusion_module import DiffusionModule 

class DDIMSampler: 
    def __init__(
        self, 
        diffusion_module: DiffusionModule, 
        num_samples: int,
        image_size: int, 
        channels: int, 
        reduce_steps: int, 
        step_collect: int, 
        device: str 
    ): 
        self.diffusion_module = diffusion_module
        self.num_samples = num_samples 
        self.image_size = image_size 
        self.channels = channels 
        self.reduce_steps = reduce_steps 
        self.step_collect = step_collect
        self.device = torch.device(device) 

        self.diffusion_model = diffusion_module.diffusion_model.to(self.device)
        self.denoise_net = self.diffusion_model.denoise_net.to(self.device)

        self.time_steps = self.diffusion_model.time_steps 
        self.betas = self.diffusion_model.betas.to(self.device) 
        self.alphas = self.diffusion_model.alphas.to(self.device)
        self.alpha_bar = self.diffusion_model.alpha_bar.to(self.device) 
        self.sqrt_alpha_bar = self.diffusion_model.sqrt_alpha_bar.to(self.device) 
        self.sqrt_one_minus_alpha_bar = self.diffusion_model.sqrt_one_minus_alpha_bar.to(self.device)

        self.tau = [i for i in range(0, self.time_steps, self.time_steps//self.reduce_steps)]
        self.tau = [i for i in reversed(self.tau)] 

    def reverse_process(self): 
        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = []
            x = torch.randn(size=(self.num_samples, self.channels, self.image_size, self.image_size), device=self.device)
            for i in range(len(self.tau)): 
                if self.tau[i] != 0: 
                    t = (torch.ones(size=self.num_samples) * self.tau[i]).long().to(self.device)
                    noise_pred = self.denoise_net(x, t)
                    alpha_bar = self.alpha_bar[t][:, None, None, None] 

                    t_prev = (torch.ones(size=self.num_samples) * self.tau[i + 1]).long().to(self.device) 
                    pred_x0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred)/torch.sqrt(alpha_bar) 
                    alpha_bar_prev = self.alpha_bar[t_prev][:, None, None, None] 

                    x = torch.sqrt(alpha_bar_prev) * pred_x0  + torch.sqrt(1 - alpha_bar_prev) * noise_pred 

                else: 
                    t = (torch.ones(size=self.num_samples) * self.tau[i]).long().to(self.device) 
                    alpha_bar = self.alpha_bar[t][:, None, None, None]
                    noise_pred = self.denoise_net(x, t) 

                    x = (x - torch.sqrt(1 - alpha_bar) * noise_pred)/torch.sqrt(alpha_bar) 
                
                collection.append(x) 
            
            return x, collection 






    


