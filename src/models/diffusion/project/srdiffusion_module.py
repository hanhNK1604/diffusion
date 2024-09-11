import lightning as L 
import torch 
from torch import nn 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchvision.utils import make_grid 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.diffusion.net.diffusion import Diffusion  # type: ignore

class SRDiffusion(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: Diffusion, 
        optimizer, 
    ): 
        super(SRDiffusion, self).__init__() 

        self.save_hyperparameters(logger=False) 
        self.diffusion_model = diffusion_model 
        self.optimizer = optimizer 

        self.psnr_metric = PeakSignalNoiseRatio() 
        self.ssim_metric = StructuralSimilarityIndexMeasure()

        self.loss_fn = nn.MSELoss() 

    def forward(self, batch): 
        hr, lr = batch
        res_image, kld_loss, feature_loss = self.diffusion_model.forward(batch) 

        loss = self.loss_fn(res_image, hr) 

        total_loss = loss + kld_loss + feature_loss 

        return total_loss, res_image 
    
    def step(self, batch): 
        total_loss, res_image = self.forward(batch) 
        return total_loss, res_image  

    def on_train_epoch_start(self): 
        self.psnr_metric.reset() 
        self.ssim_metric.reset() 
    
    def training_step(self, batch, batch_index): 
        total_loss, res_image = self.step(batch) 
        self.log('train/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 

        return total_loss 

    def validation_step(self, batch, batch_index): 
        hr, lr = batch 
        total_loss, res_image = self.step(batch) 

        ssim = self.ssim_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(res_image)) 
        psnr = self.psnr_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(res_image)) 

        self.log('val/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('val/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False) 

        if batch_index == 16: 
            hr = make_grid(self.diffusion_model.rescale(hr), nrow=2) 
            lr = make_grid(self.diffusion_model.rescale(lr), nrow=2) 
            res_image = make_grid(self.diffusion_model.rescale(res_image), nrow=2) 

            self.logger.log_image(images=[hr], key='val/hr_index_16') 
            self.logger.log_image(images=[lr], key='val/lr_index_16') 
            self.logger.log_image(images=[res_image], key='val/res_image') 
    
    def test_step(self, batch, batch_index): 
        hr, lr = batch 
        total_loss, res_image = self.step(batch) 

        ssim = self.ssim_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(res_image)) 
        psnr = self.psnr_metric(self.diffusion_model.rescale(hr), self.diffusion_model.rescale(res_image)) 

        self.log('test/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('test/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters()) 



