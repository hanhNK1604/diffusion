from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets import ImageFolder 
from torchvision import transforms
import torchvision
import os 

class CelebADataset(Dataset): 
    def __init__(
        self, 
        data_dir
    ): 
        super(CelebADataset, self).__init__() 
        self.data_dir = data_dir 

        self.data_path = os.path.join(self.data_dir, 'celeba30k') 
        self.transform_hr = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.transform_lr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.BICUBIC), 
            transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ])
        self.dataset_hr = ImageFolder(self.data_path, transform=self.transform_hr) 
        # self.dataset_lr = ImageFolder(self.data_path, transform=self.transform_lr) 
    
    def __len__(self):
        return len(self.dataset_hr) 
    
    def __getitem__(self, index): 
        return self.dataset_hr.__getitem__(index)[0] #, self.dataset_lr.__getitem__(index)[0]   
    
# data_dir = '/mnt/apple/k66/hanh/generative_AI/data' 
# dataset = CelebADataset(data_dir=data_dir) 
# print(dataset.__getitem__(0).max())
# print(dataset.__getitem__(0).min()) 
# print(dataset.__getitem__(0).shape)