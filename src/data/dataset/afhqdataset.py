from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets import ImageFolder 
from torchvision import transforms
import torchvision
import os 

class AFHQDataset(Dataset): 
    def __init__(
        self, 
        data_dir: str 
    ): 
        super(AFHQDataset, self).__init__()
        self.data_dir = data_dir 
        self.data_path = os.path.join(self.data_dir, 'afhq') 
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.dataset = ImageFolder(self.data_path, transform=self.transform) 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def __getitem__(self, index): 
        return self.dataset.__getitem__(index)[0] 
    

# data_dir = '/mnt/apple/k66/hanh/diffusion/data'
# dataset = AFHQDataset(data_dir=data_dir) 

# print(dataset.__getitem__(0).max())
# print(dataset.__getitem__(0).min()) 
# print(dataset.__getitem__(0).shape)
# print(dataset.__len__())