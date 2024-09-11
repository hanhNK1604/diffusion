import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.vae.components.Encoder import Encoder  # type: ignore
from src.models.vae.components.Decoder import Decoder  # type: ignore

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


class KLVAEModel(nn.Module): 
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        kld_weight = 1e-6
    ): 
        super(KLVAEModel, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.kld_weight = kld_weight 

    def encode(self, x): 
        mean_log_var = self.encoder.forward(x) 
        z, kld_loss = GaussianDistribution(mean_log_var).sample() 
        kld_loss = self.kld_weight * kld_loss

        return z, kld_loss

    def decode(self, z): 
        return self.decoder(z) 

    def forward(self, x): 
        z, kld_loss = self.encode(x) 
        res_image = self.decode(z) 

        return res_image, {'kld_loss': kld_loss} 

# encoder = Encoder(in_ch=3, double_latent=True).to('cuda')
# decoder = Decoder(out_ch=3).to('cuda')

# net = KLVAEModel(encoder, decoder).to('cuda') 
# a = torch.rand((1, 3, 256, 256)).to('cuda')

# res_image, kld_loss = net(a) 
# print(res_image.shape) 
# print(kld_loss)

