import torch 
from torch import nn
import torchvision

class FeatureExtractor(nn.Module): 
    def __init__(self): 
        super(FeatureExtractor, self).__init__() 
        vgg19_model = torchvision.models.vgg19(pretrained=True)  
        for param in vgg19_model.parameters(): 
            param.requires_grad = False 
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
    
    def forward(self, image): 
        return self.feature_extractor(image)