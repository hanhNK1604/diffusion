import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion 
from src.models.components.UNet import UNet 
from src.models.vae_module import VAEModule 

class LatentDiffusion(UnconditionalDiffusion): 
    def __init__(
        self, 
        vae_module_path: str, 
        denoise_net: UNet,
        time_steps: int = 1000, 
        schedule: str = 'cosine'
    ): 
        super().__init__(denoise_net=denoise_net, time_steps=time_steps, schedule=schedule)
        self.vae_module_path = vae_module_path 
        self.vae_module = VAEModule.load_from_checkpoint(vae_module_path)
        self.vae_module.eval().freeze() 

    def autuencoder_encode(self, x):
        vae_model = self.vae_module.vae_model 
        z = vae_model.encoder.forward(x) 

        return z 
    
    def autuencoder_quantize(self, z): 
        vae_model = self.vae_module.vae_model
        quantize_z, _ = vae_model.quantizer.forward(z)

        return quantize_z 

    def autoencoder_decode(self, quantize_z): 
        vae_model = self.vae_module.vae_model 
        res_image = vae_model.decoder.forward(quantize_z) 

        return res_image 

    def forward(self, batch):
        z = self.autuencoder_encode(batch)
        z = self.autuencoder_quantize(z) 
         
        self.denoise_net = self.denoise_net.to(batch.device) 

        t = torch.randint(low=0, high=self.time_steps, size=(batch.shape[0],), device=batch.device)
        noise = torch.randn_like(z, device=batch.device)             
        zt = self.forward_process(z, noise, t) 

        pred_noise = self.denoise_net.forward(zt, t) 

        return pred_noise, noise 
    




