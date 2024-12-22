import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

import lightning as L 
import torch 
from torch import nn, optim
from torch.nn import functional as F
import torchmetrics 
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.optim import Optimizer
from torchvision.utils import make_grid

from src.models.diffusion.net.srdiffusion import SuperResolutionDiffusion 
from src.models.diffusion.sampler.ddpm import DDPMSampler 
from src.models.diffusion.sampler.ddim import DDIMSampler 

class SuperResolutionDiffusionModule(L.LightningModule): 
    def __init__(
        self,
        diffusion_model: SuperResolutionDiffusion, 
        optimizer, 
        sampler: DDPMSampler
    ): 
        super(SuperResolutionDiffusionModule, self).__init__() 
        self.save_hyperparameters(logger=False)
        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 
        self.sampler = sampler 

        self.ssim = StructuralSimilarityIndexMeasure() 
        self.psnr = PeakSignalNoiseRatio() 

        self.loss_fn = nn.L1Loss() 
        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

    def forward(self, batch): 
        pred_noise, noise = self.diffusion_model.forward(batch) 
        return pred_noise, noise 
    
    def step(self, batch): 
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise) 
        return loss 
    
    def training_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss 

    def rescale(self, image): 
        return image * self.std.to(image.device) + self.mean.to(image.device)

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.sampler.denoise_net = self.diffusion_model.denoise_net 

        hr, lr = batch 
        bs = hr.shape[0] 
        minus_pred, collection = self.sampler.reverse_sr_diffusion(c=lr, batch_size=bs)
  
        minus_pred = minus_pred.clamp(-1, 1) 
        up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bilinear")
        up.required_grad = False 

        hr_pred = up + minus_pred

        hr = self.rescale(hr) 
        lr = self.rescale(lr) 
        hr_pred = self.rescale(hr_pred) 

        hr = hr.clamp(0, 1) 
        lr = lr.clamp(0, 1) 
        hr_pred = hr_pred.clamp(0, 1)

        ssim = self.ssim(hr_pred, hr) 
        psnr = self.psnr(hr_pred, hr) 

        self.log('val/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('val/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False)

        hr = make_grid(hr, nrow=2)
        lr = make_grid(lr, nrow=2) 
        hr_pred = make_grid(hr_pred, nrow=2)

    
        self.logger.log_image(images=[hr], key='val/hr_image') 
        self.logger.log_image(images=[lr], key='val/lr_image') 
        self.logger.log_image(images=[hr_pred], key='val/hr_image_reconstruct')

    def test_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.sampler.denoise_net = self.diffusion_model.denoise_net 

        hr, lr = batch 
        bs = hr.shape[0] 
        minus_pred, collection = self.sampler.reverse_sr_diffusion(c=lr, batch_size=bs)
  
        minus_pred = minus_pred.clamp(-1, 1) 
        up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bilinear")
        up.required_grad = False 

        hr_pred = up + minus_pred

        hr = self.rescale(hr) 
        lr = self.rescale(lr) 
        hr_pred = self.rescale(hr_pred) 

        hr = hr.clamp(0, 1) 
        lr = lr.clamp(0, 1) 
        hr_pred = hr_pred.clamp(0, 1)

        ssim = self.ssim(hr_pred, hr) 
        psnr = self.psnr(hr_pred, hr) 

        self.log('test/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('test/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False)

        hr = make_grid(hr, nrow=2)
        lr = make_grid(lr, nrow=2) 
        hr_pred = make_grid(hr_pred, nrow=2)

    
        self.logger.log_image(images=[hr], key='test/hr_image') 
        self.logger.log_image(images=[lr], key='test/lr_image') 
        self.logger.log_image(images=[hr_pred], key='test/hr_image_reconstruct')

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())