import zipfile 

path = '/mnt/apple/k66/hanh/diffusion/data/mri_data.zip'
path_unzip = '/mnt/apple/k66/hanh/diffusion/data/mri_data'

with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(path_unzip)