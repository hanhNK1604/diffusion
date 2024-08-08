import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
import lightning as L 
from torch import nn, optim
from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion
import torchmetrics 
from torch.optim import Optimizer

from src.models.diffusion.sampler.ddim import DDIMSampler 
from src.models.diffusion.sampler.ddpm import DDPMSampler 
from torchmetrics.image import FrechetInceptionDistance 

class DiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: UnconditionalDiffusion, 
        optimizer, 
        sampler: DDIMSampler 
    ): 
        super(DiffusionModule, self).__init__()
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer 
        self.sampler = sampler 

        self.loss_fn = nn.MSELoss() 
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, 32, 32)) 

    def forward(self, x): 
        pred_noise, noise = self.diffusion_model(x) 
        return pred_noise, noise 
    
    def step(self, batch): 
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise) 

        return loss
    
    def training_step(self, batch, batch_index):
        loss = self.step(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log('train/mae', mae, prog_bar=True, on_step=True, on_epoch=True)

        return loss 
    
    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        self.sampler.denoise_net = self.diffusion_model.denoise_net 
        bs = batch.shape[0]
        fake_images, collection = self.sampler.reverse_process(batch_size=bs) 
        real_images = batch 

        fake_images = fake_images.clamp(0, 1).repeat(1, 3, 1, 1) 
        real_images = real_images.clamp(0, 1).repeat(1, 3, 1, 1) 

        self.metric.update(fake_images, real=False) 
        self.metric.update(real_images, real=True) 
        
        fid = self.metric.compute()
        self.log('fid/val', fid, prog_bar=False) 



    def on_validation_epoch_end(self): 
        self.sampler.denoise_net = self.diffusion_model.denoise_net
        sample_images, collection = self.sampler.reverse_process(batch_size=10)  
        self.logger.log_image(key='sample images', images=[image for image in sample_images.cpu()])

        
    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
        