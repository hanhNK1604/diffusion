import torch 
from torch import nn 

class Quantizer(nn.Module): 
    def __init__(
        self, 
        num_embeds: int, 
        embed_dim: int, 
        beta: float = 0.25
    ): 
        super(Quantizer, self).__init__() 
        self.K = num_embeds 
        self.D = embed_dim 
        self.beta = beta 

        self.embedding = nn.Embedding(self.K, self.D)  
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K) 

    def forward(self, latent: torch.Tensor): 
        latent = latent.permute(0, 2, 3, 1).contiguous()
        latent_shape = latent.shape 
        flat_latent = latent.view(-1, self.D) 
        dist = torch.sum(flat_latent ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_latent, self.embedding.weight.t()) 
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1) 

        device = latent.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        quantize_latent = torch.matmul(encoding_one_hot, self.embedding.weight) 
        quantize_latent = quantize_latent.view(latent_shape)

        embedding_loss = nn.functional.mse_loss(quantize_latent.detach(), latent)
        commitment_loss = nn.functional.mse_loss(quantize_latent, latent.detach())

        vq_loss = embedding_loss + self.beta * commitment_loss 

        quantize_latent = latent + (quantize_latent - latent).detach()

        return quantize_latent.permute(0, 3, 2, 1).contiguous(), vq_loss 





# a = torch.rand((1, 3, 32, 32))
# net = Quantizer(num_embeds=512, embed_dim=3, beta=0.25) 

# z, loss = net(a) 
# print(z.shape) 
# print(loss)