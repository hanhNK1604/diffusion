import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 
import lightning as L 
import torch 
from torch import nn, optim 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure 
from torchvision.utils import make_grid 

from src.models.components.RRDB import RRDB 

class RRDBModule(L.LightningModule):
    def __init__(
        self, 
        net: RRDB, 
        optimizer
    ): 
        super(RRDBModule, self).__init__() 
        self.save_hyperparameters(logger=False)
        self.net = net 
        self.optimizer = optimizer 

        self.loss_fn = nn.L1Loss()
        self.ssim_metric = StructuralSimilarityIndexMeasure() 
        self.psnr_metric = PeakSignalNoiseRatio() 

        self.mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2)

    def forward(self, batch): 
        hr, lr = batch 
        pred_hr = self.net.forward(lr) 
        return pred_hr 

    def step(self, batch): 
        hr, lr = batch 
        pred_hr = self.forward(batch)
        loss = self.loss_fn(hr, pred_hr) 
        return loss 
    
    def rescale(self, image): 
        return image * self.std.to(image.device) + self.mean.to(image.device)
    
    def training_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss 

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=False) 

        hr, lr = batch 
        pred_hr = self.forward(batch)

        hr = self.rescale(hr) 
        lr = self.rescale(lr) 
        pred_hr = self.rescale(pred_hr) 

        ssim = self.ssim_metric(pred_hr, hr) 
        psnr = self.psnr_metric(pred_hr, hr) 

        self.log('val/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('val/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False)

        if batch_index == 0: 
            hr = make_grid(hr, nrow=2) 
            lr = make_grid(lr, nrow=2) 
            pred_hr = make_grid(pred_hr, nrow=2) 

            self.logger.log_image(images=[hr], key='val/hr_image') 
            self.logger.log_image(images=[lr], key='val/lr_image') 
            self.logger.log_image(images=[pred_hr], key='val/hr_image_reconstruct')
    
    def test_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('test/loss', loss, prog_bar=True, on_epoch=True, on_step=False) 

        hr, lr = batch 
        pred_hr = self.forward(batch)

        hr = self.rescale(hr) 
        lr = self.rescale(lr) 
        pred_hr = self.rescale(pred_hr) 

        ssim = self.ssim_metric(pred_hr, hr) 
        psnr = self.psnr_metric(pred_hr, hr) 

        self.log('test/ssim', ssim, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('test/psnr', psnr, prog_bar=True, on_epoch=True, on_step=False)
    
    def configure_optimizers(self): 
        return self.optimizer(self.parameters())

     



    