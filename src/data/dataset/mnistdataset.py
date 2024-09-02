import torch 
import os.path as osp 
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST 
from torchvision import transforms 
import rootutils 

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class MNISTDataset(Dataset): 
    def __init__(
        self, 
        image_size, 
        data_dir: str = 'data' 
    ): 
        super(MNISTDataset, self).__init__() 
        self.image_size = image_size 
        self.data_dir = osp.join(data_dir, 'mnist')
        self.data_set = None
        self.prepare_data() 

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_set = MNIST(self.data_dir, download=True, train=True, transform=transform)
        test_set = MNIST(self.data_dir, download=True, train=False, transform=transform)

        self.data_set = ConcatDataset(datasets=[train_set, test_set])

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]
    
# dataset = MNISTDataset(image_size=32)
# print(len(dataset))
# image = dataset[0]
# print(image.shape)