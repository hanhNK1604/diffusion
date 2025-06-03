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
    
    
    def pred_start_from_noise(self, x_t, t, pred_noise): 
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None] 
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None] 
        
        pred_start = 1./sqrt_alpha_bar * (x_t - sqrt_one_minus_alpha_bar * pred_noise) 
        pred_start = torch.clamp(pred_start, -1, 1) 
        return pred_start 

    def reverse_mean(self, x_t, t, pred_noise): 
        pred_start = self.pred_start_from_noise(x_t, t, pred_noise)
        
        alpha = self.alphas[t][:, None, None, None] 
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        alpha_bar_prev = self.alpha_bar[t - 1][:, None, None, None] if t[0] >= 1 else None 

        beta = self.betas[t][:, None, None, None] 


        if t[0] >= 1: 
            mean = (1. / (1 - alpha_bar)) * (torch.sqrt(alpha) * (1 - alpha_bar_prev) * x_t + torch.sqrt(alpha_bar_prev) * beta * pred_start)
        else: 
            mean = (1. / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) 
        
        return mean 

    
    def reverse_process(self, batch_size, c): 
        if batch_size is None: 
            batch_size = self.num_samples
        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = [] 
            x_t = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device) 
            for i in tqdm(reversed(range(self.time_steps))): 
                t = (torch.ones(batch_size) * i).long().to(self.device) 

                pred_noise = self.denoise_net(x_t, t, c) 
                mean = self.reverse_mean(x_t=x_t, t=t, pred_noise=pred_noise) 
                std = torch.sqrt(self.betas[t][:, None, None, None]) 

                if i >= 1: 
                    noise = torch.randn_like(x_t) 
                else: 
                    noise = torch.zeros_like(x_t) 
                
                x_t = mean + std * noise 

                if (i + 1) % self.step_collect == 0 or i == 0: 
                    collection.append(x_t) 
                    
            x_t = torch.clamp(x_t, -1, 1) 

        return x_t, collection 
        

    






    






        