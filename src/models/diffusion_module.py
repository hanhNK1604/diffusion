import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True) 

import torch 
import lightning as L 
from torch import nn, optim
from src.models.diffusion.net.unconditional_diffusion import UnconditionalDiffusion
import torchmetrics 

class DiffusionModule(L.LightningModule): 
    def __init__(
        self, 
        diffusion_model: UnconditionalDiffusion, 
        optimizer: torch.optim, 
    ): 
        super(DiffusionModule, self).__init__()
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer 

        self.loss_fn = nn.MSELoss() 
        self.metric = torchmetrics.MeanAbsoluteError() 

    def forward(self, x): 
        pred_noise, noise = self.diffusion_model(x) 
        return pred_noise, noise 
    
    def step(self, batch): 
        pred_noise, noise = self.forward(batch) 
        loss = self.loss_fn(pred_noise, noise) 
        mae = self.metric(pred_noise, noise) 

        return loss, mae 
    
    def training_step(self, batch, batch_index):
        loss, mae = self.step(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mae', mae, prog_bar=True, on_step=True, on_epoch=True)

        return loss 
    
    def validation_step(self, batch, batch_index): 
        loss, mae = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/mae', mae, prog_bar=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, mae = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", mae, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), 1e-4)