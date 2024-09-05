import torch 
from torch import nn 

class LabelEmbedding(nn.Module): 
    def __init__(
        self, 
        num_embeds: int = 10, 
        emb_dim: int = 256 
    ): 
        super(LabelEmbedding, self).__init__() 
        self.num_embeds = num_embeds 
        self.emb_dim = emb_dim 

        self.embedder = nn.Embedding(num_embeddings=num_embeds, embedding_dim=emb_dim) 

    def forward(self, x): 
        return self.embedder(x) 
