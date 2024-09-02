from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets import ImageFolder 
from torchvision import transforms
import os 

class CelebADataset(Dataset): 
    def __init__(
        self, 
        data_dir
    ): 
        super(CelebADataset, self).__init__() 
        self.data_dir = data_dir 

        self.data_path = os.path.join(self.data_dir, 'celeba30k') 
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        self.dataset = ImageFolder(self.data_path, transform=self.transform) 
    
    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index): 
        return self.dataset.__getitem__(index)[0] 
    
# data_dir = '/mnt/apple/k66/hanh/generative_AI/data' 
# dataset = CelebADataset(data_dir=data_dir) 
# print(dataset.__getitem__(0).max())
# print(dataset.__getitem__(0).min()) 
# print(dataset.__getitem__(0).shape)