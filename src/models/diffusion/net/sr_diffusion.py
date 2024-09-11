import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion 
from src.models.components.UNet import UNet 
from src.models.components.UNet_No_Attn import UNetNoAttn # type: ignore
from src.models.vae_module import VAEModule 

class SRDiffusion(UnconditionalDiffusion): 
    def __init__(
        self, 
        vae_module_path: str, 
        denoise_net: UNetNoAttn,
        time_steps: int = 200, 
        schedule: str = 'cosine'
    ): 
        super().__init__(denoise_net=denoise_net, time_steps=time_steps, schedule=schedule)
        self.vae_module_path = vae_module_path 
        self.vae_module = VAEModule.load_from_checkpoint(vae_module_path)
        self.vae_module.eval().freeze() 
        
        self.vae_model = self.vae_module.vae_model 
        for p in self.vae_model.parameters():  
            p.requires_grad = False 

        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    

    def autuencoder_encode(self, x):
        z, _ = self.vae_model.encode(x) 
        return z 

    def autoencoder_decode(self, z): 
        res_image = self.vae_model.decode(z) 
        return res_image 
    
    def rescale(self, image):
        return image * self.std.to(image.device) + self.mean.to(image.device)  

    def forward(self, batch):
        hr, lr = batch 
        self.denoise_net = self.denoise_net.to(hr.device)

        hr_latent = self.autuencoder_encode(hr)
        lr_latent = self.autuencoder_encode(lr) 
        minus_latent = hr_latent - lr_latent 

        t = torch.randint(low=0, high=self.time_steps, size=(hr.shape[0],), device=hr.device) 
        noise = torch.randn_like(minus_latent)
        minus_latent_t = self.forward_process(minus_latent, noise, t) 

        pred_noise = self.denoise_net.forward(minus_latent_t, t, lr_latent) 

        return pred_noise, noise 





