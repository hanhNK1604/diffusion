import lightning as L 
import torch 
from torch import nn 
from torchvision.utils import make_grid 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.vae.net.vq_vae import VQVAEModel # type: ignore
from src.models.vae.net.kl_vae import KLVAEModel

class VAEModule(L.LightningModule): 
    def __init__(
        self, 
        vae_model: KLVAEModel,
        optimizer
    ): 
        super(VAEModule, self).__init__()

        self.save_hyperparameters(logger=False)
        self.vae_model = vae_model 
        self.optimizer = optimizer 
        self.res_loss = nn.MSELoss()

    def forward(self, x): 
        res_image, kld_loss = self.vae_model.forward(x)
        return res_image, kld_loss


    def training_step(self, batch, batch_index): 
        res_image, kld_loss = self.forward(batch) 
        res_loss = self.res_loss(res_image, batch) 
        total_loss = res_loss + kld_loss 

        self.log('train/total_loss', total_loss, on_epoch=True, on_step=False, prog_bar=True) 
        self.log('train/res_loss', res_loss, on_epoch=True, on_step=False, prog_bar=True) 
        self.log('train/kld_loss', kld_loss, on_epoch=True, on_step=False, prog_bar=True) 

        return total_loss 

    def interpolation(self, batch): 
        
        latens, _ = self.vae_model.encode(batch) 
        steps = torch.linspace(start=0, end=1, steps=100) 

        start_latent = latens[0].unsqueeze(0)
        end_latent = latens[1].unsqueeze(0)
        minus_latent = end_latent - start_latent 

        list_latent_interpolation = [start_latent + minus_latent * i for i in steps] 
        list_decode_interpolation = [self.vae_model.decode(latent) for latent in list_latent_interpolation] 
        list_decode_interpolation = torch.cat(list_decode_interpolation, dim=0) 

        image = make_grid(list_decode_interpolation, nrow=10)

        return image 


    def validation_step(self, batch, batch_index): 
        res_image, kld_loss = self.forward(batch) 
        res_loss = self.res_loss(res_image, batch) 
        total_loss = res_loss + kld_loss 

        self.log('val/total_loss', total_loss, on_epoch=True, on_step=False, prog_bar=True) 
        self.log('val/res_loss', res_loss, on_epoch=True, on_step=False, prog_bar=True) 
        self.log('val/kld_loss', kld_loss, on_epoch=True, on_step=False, prog_bar=True) 

        if batch_index == torch.randint(low=0, high=25, size=(1,))[0]:  
            fake_image = res_image 
            real_image = batch 

            fake_image = make_grid(fake_image, nrow=2) 
            real_image = make_grid(real_image, nrow=2) 

            self.logger.log_image(images=[real_image], key='val/real_image')
            self.logger.log_image(images=[fake_image], key='val/fake_image') 

            image_interpolation = self.interpolation(batch) 

            self.logger.log_image(images=[image_interpolation], key='val/interpolation')
            
    def configure_optimizers(self): 
        return self.optimizer(self.parameters())