import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

import torch 
from torch import nn 
from src.models.vae.components.Encoder import Encoder 
from src.models.vae.components.Decoder import Decoder 
from src.models.vae.components.FeatureExtractor import FeatureExtractor 

class GaussianDistribution: 
    def __init__(
        self, 
        parameters: torch.Tensor 
    ): 
        self.mean, self.log_var = torch.chunk(parameters, chunks=2, dim=1) 
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self): 
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mean ** 2  - self.log_var.exp(), dim=[1, 2, 3]), dim=0) 
        z = self.mean + self.std * torch.randn_like(self.std) 

        return z, kld_loss 

class Diffusion(nn.Module): 
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        extractor: FeatureExtractor,
        kld_weight = 1e-6, 
        feature_weight = 0.006
    ): 
        super(Diffusion, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder 
        self.extractor = extractor 
        self.kld_weight = kld_weight
        self.feature_weight = feature_weight 
    
    def encode(self, x): 
        mean_log_var = self.encoder.forward(x) 
        z, kld_loss = GaussianDistribution(mean_log_var).sample() 

        return z, kld_loss 
    
    def decode(self, z): 
        return self.decoder.forward(z) 
    
    def extract(self, image): 
        return self.extractor.forward(image) 
    
    def rescale(self, image): 
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

        return image * std.to(image.device) + mean.to(image.device) 

    def forward(self, batch): 
        hr, lr = batch

        z, kld_loss = self.encode(lr) 
        kld_loss = self.kld_weight * kld_loss

        res_image = self.decode(z) 

        res_feature = self.extract(res_image) 
        real_feature = self.extract(hr) 

        feature_loss = self.feature_weight * torch.nn.functional.mse_loss(res_feature, real_feature) 


        return res_image, kld_loss, feature_loss 


