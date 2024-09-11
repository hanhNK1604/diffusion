import torch 
from torch import nn 

import rootutils 
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True) 

from src.models.vae.components.Encoder import Encoder  # type: ignore
from src.models.vae.components.Decoder import Decoder  # type: ignore
from src.models.vae.components.Quantizer import Quantizer  # type: ignore

class VQVAEModel(nn.Module): 
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        quantizer: Quantizer  
    ): 
        super(VQVAEModel, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder 
        self.quantizer = quantizer 

    def encode(self, x): 
        latent = self.encoder(x) 
        quantize_latent, vq_loss = self.quantizer(latent) 
        return quantize_latent, vq_loss
    

    def decode(self, quant_latent): 
        return self.decoder(quant_latent)
    
    def forward(self, x): 
        quantize_latent, vq_loss = self.encode(x) 
        res_image = self.decode(quantize_latent) 
        
        return res_image, {'vq_loss': vq_loss}


# encoder = Encoder(in_ch=3, double_latent=False).to('cuda')
# decoder = Decoder(out_ch=3).to('cuda')
# quantizer = Quantizer(num_embeds=512, embed_dim=3)


# net = VQVAEModel(encoder, decoder, quantizer).to('cuda') 
# a = torch.rand((1, 3, 256, 256)).to('cuda')

# res_image, vq_loss = net(a) 
# print(res_image.shape) 
# print(vq_loss)