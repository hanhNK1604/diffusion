import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
import lightning as L 
from torch import nn, optim
from src.models.diffusion.net.conditional_diffusion import ConditionalDiffusion
from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion
import torchmetrics 
from torch.optim import Optimizer

from torchvision.utils import make_grid

from src.models.diffusion.sampler.ddim import DDIMSampler
from src.models.diffusion.sampler.ddpm import DDPMSampler 
from torchmetrics.image import FrechetInceptionDistance 

class DiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: UnconditionalDiffusion, 
        optimizer, 
        sampler: DDPMSampler 
    ): 
        super(DiffusionModule, self).__init__()
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer 
        self.sampler = sampler 

        self.loss_fn = nn.MSELoss() 
        self.mae = torchmetrics.MeanAbsoluteError()
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, 32, 32)) 

    def forward(self, batch): 
        pred_noise, noise = self.diffusion_model(batch) 
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
    
    # def setup(self, stage: str): 
    #     if stage == "fit": 
    #         self.diffusion_model = torch.compile(self.diffusion_model) 
    
    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        x, label = batch 
        self.sampler.denoise_net = self.diffusion_model.denoise_net 
        bs = x.shape[0]
        fake_images, collection = self.sampler.reverse_process(batch_size=bs, c=None) 
        real_images = x

        fake_images = fake_images * 0.5 + 0.5 
        real_images = fake_images * 0.5 + 0.5 

        fake_images = fake_images.clamp(0, 1).repeat(1, 3, 1, 1) 
        real_images = real_images.clamp(0, 1).repeat(1, 3, 1, 1) 

        self.metric.update(fake_images, real=False) 
        self.metric.update(real_images, real=True) 
        
        fid = self.metric.compute()
        self.log('val/fid', fid, on_step=False, on_epoch=True, prog_bar=False) 


    def on_validation_epoch_end(self): 
        self.sampler.denoise_net = self.diffusion_model.denoise_net
        # label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
        grid, collection = self.sampler.reverse_process(batch_size=100, c=None) 
        
        # grid = torch.cat(grid, dim=0)
        grid = make_grid(grid, nrow=10)

        self.logger.log_image(key='val/sample_images', images=[grid])

        
    def test_step(self, batch, batch_idx):
        
        x, label = batch 
        self.sampler.denoise_net = self.diffusion_model.denoise_net 
        bs = x.shape[0]
        fake_images, collection = self.sampler.reverse_process(batch_size=bs, c=None) 
        real_images = x

        fake_images = fake_images * 0.5 + 0.5 
        real_images = fake_images * 0.5 + 0.5 

        fake_images = fake_images.clamp(0, 1).repeat(1, 3, 1, 1) 
        real_images = real_images.clamp(0, 1).repeat(1, 3, 1, 1) 

        self.metric.update(fake_images, real=False) 
        self.metric.update(real_images, real=True) 
        
        fid = self.metric.compute()
        self.log('test/fid', fid, on_step=False, on_epoch=True, prog_bar=False)  

    
    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
        