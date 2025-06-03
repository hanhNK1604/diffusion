import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
import lightning as L 
from torch import nn, optim
from src.models.diffusion.net.conditional_diffusion import ConditionalDiffusion
import torchmetrics 
from torch.optim import Optimizer

from torchvision.utils import make_grid

from src.models.diffusion.sampler.ddim import DDIMSampler 
from torchmetrics.image import FrechetInceptionDistance 
from src.models.diffusion.net.latent_diffusion import LatentDiffusion  

class LatentDiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: LatentDiffusion, 
        optimizer, 
        sampler: DDIMSampler, 
    ): 
        super(LatentDiffusionModule, self).__init__() 

        self.save_hyperparameters(logger=False)

        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 
        self.sampler = sampler 
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, 256, 256))
        self.loss_fn = nn.MSELoss() 
        

    def forward(self, batch): 
        pred_noise, noise = self.diffusion_model.forward(batch) 
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

        self.sampler.denoise_net = self.diffusion_model.denoise_net 
        z, _ = self.sampler.reverse_process(batch_size=batch.shape[0]) 
        image = self.diffusion_model.autoencoder_decode(z) 
        image = self.diffusion_model.rescale(image) 

        self.fid_metric.update(self.diffusion_model.rescale(batch), real=True) 
        self.fid_metric.update(image, real=False) 

        fid = self.fid_metric.compute() 

        self.log('val/fid', fid, on_epoch=True, on_step=False) 

        
        if batch_index == 0: 
            z, _ = self.sampler.reverse_process(batch_size=25) 
        
            image = self.diffusion_model.autoencoder_decode(z)
            image = self.diffusion_model.rescale(image) 

            image = make_grid(image, nrow=5) 

            self.logger.log_image(images=[image], key='val/sample_batch_image')


    def on_validation_epoch_end(self): 
        self.sampler.denoise_net = self.diffusion_model.denoise_net
        
        z, _ = self.sampler.reverse_process(batch_size=25) 
        
        image = self.diffusion_model.autoencoder_decode(z) 
        image = self.diffusion_model.rescale(image)

        image = make_grid(image, nrow=5) 

        self.logger.log_image(images=[image], key='val/sample_epoch_image')

    
    def test_step(self, batch, batch_index): 
        self.sampler.denoise_net = self.diffusion_model.denoise_net 
        z, _ = self.sampler.reverse_process(batch_size=batch.shape[0]) 
        image = self.diffusion_model.autoencoder_decode(z) 
        image = self.diffusion_model.rescale(image) 

        self.fid_metric.update(self.diffusion_model.rescale(batch), real=True) 
        self.fid_metric.update(image, real=False) 

        fid = self.fid_metric.compute() 

        self.log('test/fid', fid, on_epoch=True, on_step=False) 

    
    