import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

import torch 
from tqdm import tqdm 


from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion 

class DDIMSampler: 
    def __init__(
        self, 
        diffusion_model: UnconditionalDiffusion, 
        num_samples: int,
        image_size: int, 
        channels: int, 
        reduce_steps: int,  
        device: str 
    ): 
        self.diffusion_model = diffusion_model.to(torch.device(device))
        self.num_samples = num_samples 
        self.image_size = image_size 
        self.channels = channels 
        self.reduce_steps = reduce_steps 
        self.device = torch.device(device) 

        self.denoise_net = self.diffusion_model.denoise_net.to(self.device)

        self.time_steps = self.diffusion_model.time_steps 
        self.betas = self.diffusion_model.betas.to(self.device) 
        self.alphas = self.diffusion_model.alphas.to(self.device)
        self.alpha_bar = self.diffusion_model.alpha_bar.to(self.device) 
        self.sqrt_alpha_bar = self.diffusion_model.sqrt_alpha_bar.to(self.device) 
        self.sqrt_one_minus_alpha_bar = self.diffusion_model.sqrt_one_minus_alpha_bar.to(self.device)

        self.tau = [i for i in range(0, self.time_steps, self.time_steps//self.reduce_steps)]
        self.tau = [i for i in reversed(self.tau)] 

    def reverse_process(self, batch_size=None, c=None): 
        if batch_size == None:
            batch_size = self.num_samples

        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = []
            x = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device)
            for i in range(len(self.tau)): 
                if self.tau[i] != 0: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device)
                    noise_pred = self.denoise_net(x, t, c)
                    alpha_bar = self.alpha_bar[t][:, None, None, None] 

                    t_prev = (torch.ones(size=(batch_size,)) * self.tau[i + 1]).long().to(self.device) 
                    pred_x0 = (x - torch.sqrt(1. - alpha_bar) * noise_pred)/torch.sqrt(alpha_bar) 
                    pred_x0 = torch.clamp(pred_x0, -1, 1)
                    alpha_bar_prev = self.alpha_bar[t_prev][:, None, None, None] 

                    x = torch.sqrt(alpha_bar_prev) * pred_x0  + torch.sqrt(1 - alpha_bar_prev) * noise_pred 

                else: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device) 
                    alpha_bar = self.alpha_bar[t][:, None, None, None]
                    noise_pred = self.denoise_net(x, t, c) 

                    x = (x - torch.sqrt(1 - alpha_bar) * noise_pred)/torch.sqrt(alpha_bar) 
                
                x = torch.clamp(x, min=-1, max=1)
                collection.append(x) 
            
            return x, collection 
    

    def reverse_process_condition(self, w, c, batch_size=None): 
        w = torch.tensor([w]).to(self.device)
        c = c.to(self.device) 

        if batch_size == None: 
            batch_size = self.num_samples 

        self.denoise_net.eval()
        with torch.no_grad(): 
            collection = [] 
            x = torch.randn(size=(batch_size, self.channels, self.image_size, self.image_size), device=self.device) 
            for i in range(len(self.tau)): 
                if self.tau[i] != 0: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device)
                    t_prev = (torch.ones(size=(batch_size,)) * self.tau[i + 1]).long().to(self.device)

                    pred_noise_no_cond = self.denoise_net(x, t)    
                    pred_noise_with_cond = self.denoise_net(x, t, c=c)
                    final_noise_pred = (1 + w) * pred_noise_with_cond - w * pred_noise_no_cond 

                    alpha_bar = self.alpha_bar[t][:, None, None, None ]
                    alpha_bar_prev = self.alpha_bar[t_prev][:, None, None, None] 

                    pred_x0 = (x - torch.sqrt(1. - alpha_bar) * final_noise_pred)/torch.sqrt(alpha_bar) 
                    pred_x0 = torch.clamp(pred_x0, -1, 1); 

                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * final_noise_pred
                
                else: 
                    t = (torch.ones(size=(batch_size,)) * self.tau[i]).long().to(self.device)

                    pred_noise_no_cond = self.denoise_net(x, t, c=None)  
                    pred_noise_with_cond = self.denoise_net(x, t, c=c)
                    final_noise_pred = (1 + w) * pred_noise_with_cond - w * pred_noise_no_cond 

                    alpha_bar = self.alpha_bar[t][:, None, None, None ]
                    alpha_bar_prev = self.alpha_bar[t_prev][:, None, None, None] 

                    x = (x - torch.sqrt(1. - alpha_bar) * final_noise_pred)/torch.sqrt(alpha_bar)
                
                collection.append(x) 

            return x, collection 
                    









    


