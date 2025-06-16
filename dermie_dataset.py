import torch
import os
from torch.utils.data import Dataset, WeightedRandomSampler  
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd


def clean_metadata(pd_metadata: pd.DataFrame, images_dir):

    # Remove Unnamed Columns
    pd_metadata = pd_metadata[['Image_id', '15_diseases', 'Fitzpatrick']]

    # Change Column Name
    pd_metadata.rename(columns={'15_diseases': 'Diagnosis'}, inplace=True)
    pd_metadata.rename(columns={'Image_id': 'Image Name'}, inplace=True)
   
    # Remove Unlabelled FST
    pd_metadata = pd_metadata[pd_metadata['Fitzpatrick'] != 'TODO']

    # Remove Unlabelled Diagnsois
    pd_metadata.dropna(subset=['Diagnosis'], inplace=True)

    # Remove Duplicates
    pd_metadata.drop_duplicates(subset=['Image Name'], inplace=True)

    # Remove Inexistent Images
    base_name_to_file = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(images_dir)
        if not f.lower().endswith('.avif')
    }
    pd_metadata = pd_metadata[pd_metadata['Image Name'].isin(base_name_to_file.keys())]
    pd_metadata['Image Name'] = pd_metadata['Image Name'].map(base_name_to_file)

    # Stratification Column
    pd_metadata['stratify_col'] = pd_metadata['Diagnosis'].astype(str) + '_' + pd_metadata['Fitzpatrick'].astype(str)

    return pd_metadata


class DermieDataset(Dataset):
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
        self.stratify_col = self.metadata['stratify_col'].values
        
        self.img_ids = self.metadata['Image Name'].values
        
        fst_encoder = OrdinalEncoder(categories=[['I', 'II', 'III', 'IV', 'V', 'VI']])
        self.fst_labels = fst_encoder.fit_transform(self.metadata['Fitzpatrick'].values.reshape(-1, 1)) + 1
        
        self.condition = self.metadata['Diagnosis'].values
        if diagnostic_encoder is None:
            self.diagnose_encoder = OneHotEncoder()
            self.diagnostic = self.diagnose_encoder.fit_transform(self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
        
        else:
            self.diagnose_encoder = diagnostic_encoder
            self.diagnostic = self.diagnose_encoder.transform(self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
            
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
        labels = dataset.metadata['Diagnosis'].astype('category').cat.codes.values
    elif choice == 'fst':
        labels = dataset.fst_labels
    elif choice == 'both':
        labels = dataset.stratify_col

    labels = np.array(labels)
    class_sample_counts = np.bincount(labels)
    class_weights = 1. / class_sample_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),  
    replacement=True)  

    return sampler


