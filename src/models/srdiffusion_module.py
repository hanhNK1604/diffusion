import lightning as L 
import torch 
from torch import nn 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchvision.utils import make_grid 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.diffusion.net.diffusion import Diffusion  # type: ignore
from src.models.diffusion.net.sr_diffusion import SRDiffusion # type: ignore 
from src.models.diffusion.sampler.ddpm import DDPMSampler 

class SRDiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: SRDiffusion, 
        sampler: DDPMSampler, 
        optimizer, 
    ): 
        super(SRDiffusionModule, self).__init__() 

        self.save_hyperparameters(logger=False) 
        self.diffusion_model = diffusion_model 
        self.sampler = sampler 
        self.optimizer = optimizer 

        self.psnr_metric = PeakSignalNoiseRatio() 
        self.ssim_metric = StructuralSimilarityIndexMeasure()

        self.loss_fn = nn.MSELoss() 

    def forward(self, batch): 
        pred_noise, noise = self.diffusion_model.forward(batch)
        loss = self.loss_fn(pred_noise, noise) 

        return loss 
    
    def step(self, batch): 
        loss = self.forward(batch) 
        return loss  

    def on_train_epoch_start(self): 
        self.psnr_metric.reset() 
        self.ssim_metric.reset() 
    
    def training_step(self, batch, batch_index): 
        total_loss = self.step(batch) 
        self.log('train/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 

        return total_loss 

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/total_loss', loss, on_epoch=True, on_step=False)

        hr, lr = batch 
        hr_latent = self.diffusion_model.autuencoder_encode(hr)
        lr_latent = self.diffusion_model.autuencoder_encode(lr) 

        self.sampler.denoise_net = self.diffusion_model.denoise_net 

        if batch_index == 16: 
            minus_latent_pred, _ = self.sampler.reverse_process_no_guidance(c=lr_latent, batch_size=hr_latent.shape[0]) 
            hr_pred = self.diffusion_model.autoencoder_decode(minus_latent_pred + lr_latent) 

            self.logger.log_image(images=[make_grid(self.diffusion_model.rescale(hr_pred), nrow=2)], key='val/hr_pred') 
            self.logger.log_image(images=[make_grid(self.diffusion_model.rescale(hr), nrow=2)], key='val/hr_image')
            self.logger.log_image(images=[make_grid(self.diffusion_model.rescale(lr), nrow=2)], key='val/lr_image')
    
    def test_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('test/total_loss', loss, on_epoch=True, on_step=False)

        hr, lr = batch 
        hr_latent = self.diffusion_model.autuencoder_encode(hr)
        lr_latent = self.diffusion_model.autuencoder_encode(lr) 

        self.sampler.denoise_net = self.diffusion_model.denoise_net 

        minus_latent_pred, _ = self.sampler.reverse_process_no_guidance(c=lr_latent, batch_size=hr_latent.shape[0]) 
        hr_pred = self.diffusion_model.autoencoder_decode(minus_latent_pred + lr_latent) 

        psnr = self.psnr_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(hr_pred))
        ssim = self.ssim_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(hr_pred))

        self.log('test/psnr', psnr, on_epoch=True, on_step=False)
        self.log('test/ssim', ssim, on_epoch=True, on_step=False)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters()) 



