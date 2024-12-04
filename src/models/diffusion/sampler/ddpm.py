import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
from tqdm import tqdm 
from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion 

class DDPMSampler: 
    def __init__(
        self, 
        diffusion_model: UnconditionalDiffusion, 
        num_samples: int,
        image_size: int, 
        channels: int, 
        step_collect: int, 
        device: str 
    ): 
        self.diffusion_model = diffusion_model.to(torch.device(device))
        self.num_samples = num_samples 
        self.image_size = image_size 
        self.channels = channels 
        self.step_collect = step_collect
        self.device = torch.device(device) 

        self.denoise_net = self.diffusion_model.denoise_net.to(self.device)

        self.time_steps = self.diffusion_model.time_steps 
        self.betas = self.diffusion_model.betas.to(self.device) 
        self.alphas = self.diffusion_model.alphas.to(self.device)
        self.alpha_bar = self.diffusion_model.alpha_bar.to(self.device) 
        self.sqrt_alpha_bar = self.diffusion_model.sqrt_alpha_bar.to(self.device) 
        self.sqrt_one_minus_alpha_bar = self.diffusion_model.sqrt_one_minus_alpha_bar.to(self.device)
    
    def reverse_process(self, batch_size=None):
        if batch_size == None: 
            batch_size = self.num_samples 
        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = []
            x = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device)
            for i in tqdm(reversed(range(self.time_steps))): 
                t = (torch.ones(batch_size) * i).long().to(self.device) 

                pred_noise = self.denoise_net(x, t) 
                alphas = self.alphas[t][:, None, None, None] 
                alpha_bar = self.alpha_bar[t][:, None, None, None] 
                betas = self.betas[t][:, None, None, None] 

                mean = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * pred_noise) 
                std = torch.sqrt(betas) 

                if i >= 1: 
                    noise = torch.randn_like(x) 
                else: 
                    noise = torch.zeros_like(x) 
                
                x = mean + std * noise 

                if (i + 1) % self.step_collect == 0 or i == 0: 
                    collection.append(x)
        
        return x, collection 
    
    def reverse_process_no_guidance(self, c, batch_size=None):
        if batch_size == None: 
            batch_size = self.num_samples 
        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = []
            x = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device)
            for i in tqdm(reversed(range(self.time_steps))): 
                t = (torch.ones(batch_size) * i).long().to(self.device) 

                pred_noise = self.denoise_net(x, t, c) 
                
                alphas = self.alphas[t][:, None, None, None] 
                alpha_bar = self.alpha_bar[t][:, None, None, None] 
                betas = self.betas[t][:, None, None, None] 

                mean = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * pred_noise)
                std = torch.sqrt(betas) 

                if i >= 1: 
                    noise = torch.randn_like(x) 
                else: 
                    noise = torch.zeros_like(x) 
                
                x = mean + std * noise 

                if (i + 1) % self.step_collect == 0 or i == 0: 
                    collection.append(x)
            
        
        return x, collection 
    






        