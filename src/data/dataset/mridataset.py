import torch 
import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms 
import rootutils 
import pandas as pd 
from PIL import Image 
import json 
import random 

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class MRIDataset(Dataset): 
    def __init__(
        self, 
        image_size, 
        data_dir: str = "data", 
        type_plane: str = "axial", 
        max_age: float = 100.0
    ): 
        super(MRIDataset, self).__init__() 
        self.image_size = image_size 
        self.type_plane = type_plane
        self.data_dir = os.path.join(data_dir, "mri_data", self.type_plane) 
        self.label_dir = os.path.join(data_dir, "mri_data", "id_age.json") 

        self.max_age = max_age

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),              
            transforms.Grayscale(num_output_channels=1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.list_full_file_name = os.listdir(self.data_dir) 
        self.list_id = [os.path.splitext(file_name)[0] for file_name in self.list_full_file_name]

        with open(self.label_dir, "r") as f:
            self.id_age_map = json.load(f)

    def __len__(self): 
        return len(self.list_id) 
    
    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.list_full_file_name[index]) 
        image = Image.open(path) 
        image = self.transform(image) 
        label = torch.tensor(
            (self.id_age_map[self.list_id[index]] + random.uniform(-1, 1)) / self.max_age,
            dtype=torch.float32
        )

        return image, label 
    

# dataset = MRIDataset(image_size=128, data_dir="/mnt/apple/k66/hanh/generative_model/data") 
# print(dataset[0][0].shape, dataset[0][1])
# print(dataset[0][0].max(), dataset[0][0].min())