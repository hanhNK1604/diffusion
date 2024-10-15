import zipfile 

path = '/mnt/apple/k66/hanh/diffusion/data/afhq.zip'
path_unzip = '/mnt/apple/k66/hanh/diffusion/data/afhq'

with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(path_unzip)