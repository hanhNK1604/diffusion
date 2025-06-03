import rootutils 
rootutils.setup_root(search_from=__file__, indicator='.project-root', pythonpath=True)  

import torch
import lightning as L 
from torch import nn, optim 
import torchmetrics 
from torch.optim import Optimizer
from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance 

from flow_matching.solver import ODESolver 
from flow_matching.path import CondOTProbPath 
from flow_matching.utils.model_wrapper import ModelWrapper 

from src.models.flow_matching.flow_matching import VelocityModel 
from diffusers.models import AutoencoderKL 

class Flow(ModelWrapper): 
    def __init__(self, net: nn.Module, w: float=4.0): 
        super(Flow, self).__init__(model=net) 
        self.w = w 

    def forward(self, x, t, **extras):
        c = extras.get("c") 
        if c is None: 
            return self.model.forward(x=x, t=t, c=None) 
        else: 
            return (1 - self.w) * self.model.forward(x=x, t=t, c=None) + self.w * self.model.forward(x=x, t=t, c=c) 
            
class FlowMatchingModule(L.LightningModule): 
    def __init__(self, 
        num_steps:int, 
        velocity_model: VelocityModel, 
        optimizer: Optimizer, 
        use_condition: bool = False,
        w: float = 4.0
    ): 
        super(FlowMatchingModule, self).__init__()  
        self.save_hyperparameters(logger=False)
        self.num_steps = num_steps
        self.velocity_model = velocity_model
        self.optimizer = optimizer 
        self.use_condition = use_condition 
        self.w = w 

        self.loss_fn = nn.MSELoss() 
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, 32, 32)) 

        self.model_wrapper = Flow(net=self.velocity_model, w=self.w) 
        self.solver = ODESolver(velocity_model=self.model_wrapper)
        self.path = CondOTProbPath()
        
        
    def forward(self, x, t, c=None): 
        velocity = self.velocity_model.forward(x=x, t=t, c=c) 
        return velocity 

    def sample(self, input_size: torch.Size, c=None): 
        noise = torch.randn(size=input_size, device="cuda") 
        extras = {"c": c} 
        fake_image = self.solver.sample(x_init=noise, step_size=self.num_steps, method="midpoint", **extras) 

        return fake_image 
        

    def step(self, batch): 
        x, c = batch 
        t = torch.rand(size=(x.shape[0],), device=x.device) 
        noise = torch.randn_like(x, device=x.device) 

        path_sample = self.path.sample(x_0=noise, x_1=x, t=t)
        x_t = path_sample.x_t.to(x.device) 
        dx_t = path_sample.dx_t.to(x.device)

        velocity = None 
        
        if not self.use_condition: 
            velocity = self.forward(x=x_t, t=t, c=None) 
        else: 
            rand_value = torch.rand(size=(1,)) 
            if rand_value[0] > 0.5: 
                velocity = self.forward(x=x_t, t=t, c=c) 
            else: 
                velocity = self.forward(x=x_t, t=t, c=None) 

        loss = self.loss_fn(velocity, dx_t) 
        return loss 

    def training_step(self, batch, batch_index): 
        loss = self.step(batch=batch) 
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True) 

        return loss 

    def validation_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        x, label = batch 
        batch_size = x.shape[0]
        real_images = x
        fake_images = None
        
        if not self.use_condition: 
            fake_images = self.sample(input_size=real_images.shape, c=None) 
        else: 
            fake_images = self.sample(input_size=real_images.shape, c=label) 

        fake_images = fake_images * 0.5 + 0.5 
        real_images = fake_images * 0.5 + 0.5 

        fake_images = fake_images.clamp(0, 1).repeat(1, 3, 1, 1) 
        real_images = real_images.clamp(0, 1).repeat(1, 3, 1, 1) 

        self.metric.update(fake_images, real=False) 
        self.metric.update(real_images, real=True) 
        
        fid = self.metric.compute()
        self.log('val/fid', fid, on_step=False, on_epoch=True, prog_bar=False) 
    
    def on_validation_epoch_end(self): 

        if not self.use_condition: 
            noise = torch.randn(size=(100, 1, 32, 32)).to("cuda")  

            grid = self.solver.sample(x_init=noise, step_size=1./self.num_steps, method="midpoint")

            grid = make_grid(grid, nrow=10)
            grid = grid * 0.5 + 0.5 

            self.logger.log_image(key='val/sample_images', images=[grid])
        
        else:
            c = (torch.randint(low=18, high=100, size=(4,)).float()/100.0).float().to("cuda") 
            list_image = self.sample(input_size=torch.Size([4, 1, 128, 128], c=c))

            grid = make_grid(list_image, nrow=2)
            grid = grid * 0.5 + 0.5 

            self.logger.log_image(key='val/sample_images', images=[grid])

    
    def test_step(self, batch, batch_index): 
        loss = self.step(batch) 
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        x, label = batch 
        real_images = x
        fake_images = None
        
        if not self.use_condition: 
            fake_images = self.sample(input_size=real_images.shape, c=None) 
        else: 
            fake_images = self.sample(input_size=real_images.shape, c=label) 

        fake_images = fake_images * 0.5 + 0.5 
        real_images = fake_images * 0.5 + 0.5 

        fake_images = fake_images.clamp(0, 1).repeat(1, 3, 1, 1) 
        real_images = real_images.clamp(0, 1).repeat(1, 3, 1, 1) 

        self.metric.update(fake_images, real=False) 
        self.metric.update(real_images, real=True) 
        
        fid = self.metric.compute()
        self.log('test/fid', fid, on_step=False, on_epoch=True, prog_bar=False) 


    def configure_optimizers(self):
        return self.optimizer(self.parameters()) 

        
