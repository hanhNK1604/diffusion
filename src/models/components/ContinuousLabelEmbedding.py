import torch 
from torch import nn 

class ContinuousLabelEmbedding(nn.Module): 
    def __init__(
        self, 
        emb_dim: int = 256 
    ): 
        super(ContinuousLabelEmbedding, self).__init__() 
        self.emb_dim = emb_dim 

        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=emb_dim), 
            nn.SiLU() 
        )
    

    def forward(self, x): 
        x = x.view(-1, 1) 
        return self.net(x) 
