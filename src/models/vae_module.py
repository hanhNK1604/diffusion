import lightning as L 
import torch 
from torch import nn 
from torchvision.utils import make_grid 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.vae.net.vq_vae import VQVAEModel 
from src.models.vae.net.kl_vae import KLVAEModel
from src.models.vae.components.FeatureExtractor import FeatureExtractor # type: ignore

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
        self.feature_extractor = FeatureExtractor()

        self.res_loss = nn.MSELoss()
        self.feature_loss = nn.MSELoss()

        self.psnr_metric = PeakSignalNoiseRatio() 
        self.ssim_metric = StructuralSimilarityIndexMeasure()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        

    def get_feture_loss(self, real, fake): 
        real_feature = self.feature_extractor.forward(real) 
        fake_feature = self.feature_extractor.forward(fake)

        return 0.006 * self.feature_loss(real_feature, fake_feature)

    def rescale(self, image):
        return image * self.std.to(image.device) + self.mean.to(image.device) 


    def forward(self, x): 
        res_image, losses = self.vae_model.forward(x) #losses only contain kld_loss 

        feature_loss = self.get_feture_loss(res_image, x) 
        res_loss = self.res_loss(res_image, x) 

        losses['feature_loss'] = feature_loss 
        losses['res_loss'] = res_loss 

        return res_image, losses


    def training_step(self, batch, batch_index): 
        res_image, losses = self.forward(batch) 
        
        total_loss = sum(losses.values())
        self.log('train/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 

        for key in losses.keys(): 
            self.log(f'train/{key}', losses[key].detach(), on_step=False, on_epoch=True) 
        
        return total_loss

    def interpolation(self, batch): 
        
        latens, _ = self.vae_model.encode(batch) 
        steps = torch.linspace(start=0, end=1, steps=100) 

        start_latent = latens[0].unsqueeze(0)
        end_latent = latens[1].unsqueeze(0)
        minus_latent = end_latent - start_latent 

        list_latent_interpolation = [start_latent + minus_latent * i for i in steps] 
        list_decode_interpolation = [self.rescale(self.vae_model.decode(latent)) for latent in list_latent_interpolation] 
        list_decode_interpolation = torch.cat(list_decode_interpolation, dim=0) 

        image = make_grid(list_decode_interpolation, nrow=10)

        return image 


    def validation_step(self, batch, batch_index): 
        res_image, losses = self.forward(batch) 

        ssim = self.ssim_metric(self.rescale(res_image), self.rescale(batch)) 
        psnr = self.psnr_metric(self.rescale(res_image), self.rescale(batch)) 

        self.log('val/ssim', ssim, on_epoch=True, on_step=False) 
        self.log('val/psnr', psnr, on_epoch=True, on_step=False) 
        
        total_loss = sum(losses.values())
        self.log('val/total_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False) 
        
        for key in losses.keys(): 
            self.log(f'val/{key}', losses[key].detach(), on_step=False, on_epoch=True) 

        if batch_index == 16:  
            fake_image = self.rescale(res_image) 
            real_image = self.rescale(batch) 

            fake_image = make_grid(fake_image, nrow=2) 
            real_image = make_grid(real_image, nrow=2) 

            self.logger.log_image(images=[real_image], key='val/real_image')
            self.logger.log_image(images=[fake_image], key='val/fake_image') 

            image_interpolation = self.interpolation(batch) 

            self.logger.log_image(images=[image_interpolation], key='val/interpolation')
        
    def test_step(self, batch, batch_index): 
        res_image, losses = self.forward(batch) 

        ssim = self.ssim_metric(self.rescale(res_image), self.rescale(batch)) 
        psnr = self.psnr_metric(self.rescale(res_image), self.rescale(batch)) 

        self.log('test/ssim', ssim, on_epoch=True, on_step=False) 
        self.log('test/psnr', psnr, on_epoch=True, on_step=False) 
            
    def configure_optimizers(self): 
        return self.optimizer(self.parameters())