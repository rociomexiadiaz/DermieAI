import torch
import os
from torch.utils.data import Dataset, WeightedRandomSampler    
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter


class PADUFESDataset(Dataset):
    def __init__(self, pd_metadata, images_dir, transform=None, diagnostic_encoder=None):
        """
        Args:
            pd_metadata (pd DataFrame): metadata dataframe.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd_metadata
        self.images_dir = images_dir
        self.transform = transform

        self.img_ids = self.metadata['img_id'].values
        self.fst_labels = self.metadata['fitspatrick'].values  
        self.condition = self.metadata['diagnostic'].values
        if diagnostic_encoder is None:
            self.diagnose_encoder = OneHotEncoder()
            self.diagnostic = self.diagnose_encoder.fit_transform(self.metadata['diagnostic'].values.reshape(-1, 1)).toarray()  

        else:
            self.diagnose_encoder = diagnostic_encoder
            self.diagnostic = self.diagnose_encoder.transform(self.metadata['diagnostic'].values.reshape(-1, 1)).toarray()
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = f"{self.img_ids[idx]}"  
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        
        fst = torch.tensor(self.fst_labels[idx], dtype=torch.float)
        diagnosis = torch.tensor(self.diagnostic[idx], dtype=torch.float)
        
        sample = {
            'image': image,
            'img_id': self.img_ids[idx],
            'fst': fst,
            'diagnosis': diagnosis,
            'condition': self.condition[idx]
        }
        
        return sample



def BalanceSampler(dataset, choice='diagnostic'):

    if choice == 'diagnostic':
        labels = dataset.metadata['diagnostic'].astype('category').cat.codes.values
    elif choice == 'fst':
        labels = dataset.fst_labels

    labels = np.array(labels)
    class_sample_counts = np.bincount(labels)
    class_weights = 1. / class_sample_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),  
    replacement=True)  

    return sampler

def DoubleBalanceSampler(dataset):

    fst = dataset.fst_labels  # assumed to be integers
    diagnostic = dataset.metadata['diagnostic'].astype('category').cat.codes.values

    joint_labels = list(zip(fst, diagnostic))
    joint_counts = Counter(joint_labels)

    sample_weights = np.array([1.0 / joint_counts[pair] for pair in joint_labels])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
