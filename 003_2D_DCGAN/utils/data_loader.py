
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio
import os




class_list = ['baseline', 'dense_cluster', 'loose_cluster']
class_num = len(class_list)


transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5, 0.5), std=(0.5,0.5, 0.5))
])



def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_



class BMData(Dataset):
    def __init__(self, data_path, img_folder, n_rows = None, transform=None):
        self.transform = transform
        if n_rows is None:
            df = pd.read_csv(data_path)
            min_ = df['size'].min()
            min_ = df.agg({'size': percentile(0.01)}).values[0]
            df = df.query(f'size > {min_}')
        else:
            df=pd.read_csv(data_path)
            min_ = df['size'].min()
            min_ = df.agg({'size': percentile(0.01)}).values[0]
            df = df.query(f'size > {min_}')
            df = df.groupby('cluster_group').apply(lambda x: x.sample(n_rows)).reset_index(drop=True)
            
        self.images = np.asarray([imageio.imread(os.path.join(img_folder, x)) for x in df.file])
        self.labels = df.cluster_group.values.astype(int)
        print('Image size:', self.images.shape)
        print('--- Label ---')
        print(df.cluster_group.astype(int).value_counts())

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.images[idx]
        img = Image.fromarray(self.images[idx])
            
        if self.transform:
            img = self.transform(img)
        
        return img, label