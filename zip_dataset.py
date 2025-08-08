import torch
import os
from torch.utils.data import Dataset, WeightedRandomSampler  
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import open_clip
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_metadata(pd_metadata: pd.DataFrame, images_dir):

    if '15_diseases' in pd_metadata.columns and 'Image_id' in pd_metadata.columns:
        
        # Remove Unnamed Columns
        pd_metadata = pd_metadata[['Image_id', '15_diseases', 'Fitzpatrick']]

        # Change Column Name
        pd_metadata = pd_metadata.rename(columns={'15_diseases': 'Diagnosis'})
        pd_metadata = pd_metadata.rename(columns={'Image_id': 'Image Name'})
   
    # Remove Unlabelled FST
    pd_metadata = pd_metadata[pd_metadata['Fitzpatrick'] != 'TODO']
    pd_metadata = pd_metadata[pd_metadata['Fitzpatrick'] != 'NONE_IDENTIFIED']
    pd_metadata = pd_metadata.dropna(subset=['Fitzpatrick'])

    # Remove Unlabelled Diagnsois
    pd_metadata.dropna(subset=['Diagnosis'], inplace=True)

    # Remove Duplicates
    pd_metadata.drop_duplicates(subset=['Image Name'], inplace=True)

    # Ensure Image Name doesn't contain .png or .jpg suffixes
    pd_metadata['Image Name'] = pd_metadata['Image Name'].apply(lambda x:os.path.splitext(x)[0])

    # Remove Inexistent Images
    with zipfile.ZipFile(images_dir, 'r') as zip_ref:
        base_name_to_file = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in zip_ref.namelist()
            if not f.lower().endswith('.avif')
        }

    pd_metadata = pd_metadata[pd_metadata['Image Name'].isin(base_name_to_file.keys())]
    pd_metadata['Image Name'] = pd_metadata['Image Name'].map(base_name_to_file)

    # Stratification Column
    pd_metadata['stratify_col'] = pd_metadata['Diagnosis'].astype(str) + '_' + pd_metadata['Fitzpatrick'].astype(str)

    return pd_metadata


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


class MultipleDatasets(Dataset):
    def __init__(self, metadata_list, image_zip_list, transform=None, diagnostic_encoder=None, clip=False, apply_augment=False):
        """
        Args:
            metadata_list (list of pd.DataFrame): List of metadata DataFrames.
            image_zip_list (list of str): List of paths to image ZIP files (same order as metadata).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.clip = clip
        self.apply_augment = apply_augment

        assert len(metadata_list) == len(image_zip_list), "Metadata and image zip lists must be the same length"
        
        valid_pairs = [(metadata, zip_path) for metadata, zip_path in zip(metadata_list, image_zip_list) 
                            if metadata is not None and zip_path is not None]
        
        if not valid_pairs:
            raise ValueError("All datasets are None! No valid datasets to load.")
        
        # Separate back into lists
        valid_metadata_list = [pair[0] for pair in valid_pairs]
        valid_image_zip_list = [pair[1] for pair in valid_pairs]

        self.transform = transform
        self.zip_paths = valid_image_zip_list
        self.zips = [zipfile.ZipFile(p, 'r') for p in valid_image_zip_list]  
        self.metadata = pd.DataFrame()

        for i, df in enumerate(valid_metadata_list):
            df = df.copy()
            df['zip_index'] = i  
            self.metadata = pd.concat([self.metadata, df], ignore_index=True)

        self.img_ids = self.metadata['Image Name'].values
        self.zip_indices = self.metadata['zip_index'].values
        self.stratify_col = self.metadata['stratify_col'].values

        # Fitzpatrick encoding
        fst_encoder = OrdinalEncoder(categories=[['I', 'II', 'III', 'IV', 'V', 'VI']])
        self.fst_labels = fst_encoder.fit_transform(self.metadata['Fitzpatrick'].values.reshape(-1, 1)) + 1

        self.condition = self.metadata['Diagnosis'].values

        if diagnostic_encoder is None:
            self.diagnose_encoder = OneHotEncoder()
            self.diagnostic = self.diagnose_encoder.fit_transform(
                self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
        else:
            self.diagnose_encoder = diagnostic_encoder
            self.diagnostic = self.diagnose_encoder.transform(
                self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
            
        if self.clip:
            self.lesion_clip_model, _, self.lesion_clip_preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
            global device
            self.lesion_clip_model.to(device)
            self.lesion_clip_model.eval()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_ids[idx]
        zip_idx = self.zip_indices[idx]
        zip_file = self.zips[zip_idx]

        try:
            with zip_file.open(img_name) as file:
                image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name} from zip {self.zip_paths[zip_idx]}: {e}")
            image = Image.new('RGB', (224, 224), color='white')


        if self.clip: 
            if self.apply_augment:
                transformations = transforms.Compose([
                        transforms.RandomAffine(degrees=10, shear= (-10,10,-10,10))])
                image = transformations(image)

            image = self.lesion_clip_preprocess(image).unsqueeze(0).to(device)
    
            with torch.no_grad():
                image = self.lesion_clip_model.encode_image(image)
        
            image /= image.norm(dim=-1, keepdim=True)

        else:
            if self.transform:
                image = self.transform(image)

        fst = torch.tensor(self.fst_labels[idx], dtype=torch.float)
        diagnosis = torch.tensor(self.diagnostic[idx], dtype=torch.float)

        return {
            'image': image,
            'img_id': img_name,
            'fst': fst,
            'diagnosis': diagnosis,
            'condition': self.condition[idx]
        }
    

def visualise(dataset: MultipleDatasets):

    metadata = dataset.metadata
    print(metadata['Diagnosis'].value_counts(dropna=False))

    fst_color_map = {
    'I': '#F5D5A0',
    'II': '#E4B589',
    'III': '#D1A479',
    'IV': '#C0874F',
    'V': '#A56635',
    'VI': '#4C2C27'
    }

    n_conditions = len(metadata['Diagnosis'].unique())

    # Calculate grid dimensions
    n_cols = 3  # Adjust as needed
    n_rows = int(np.ceil(n_conditions / n_cols))

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, condition in enumerate(metadata['Diagnosis'].unique()):
        counts = metadata[metadata['Diagnosis'] == condition]['Fitzpatrick'].value_counts().sort_index()
        colors = [fst_color_map[fst] for fst in counts.index]
        
        # Plot on the specific subplot
        axes[i].pie(counts.values, labels=counts.index, colors=colors, 
                    autopct='%1.1f%%', startangle=90)
        axes[i].set_title(condition)
        axes[i].axis('equal')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig


def load_dataset(project_dir, path_folder, images_dir, metadata_dir, stratification_strategy, seed):
    path = os.path.join(project_dir, rf'{path_folder}')
    images = rf'{path}/{images_dir}'
    metadata = clean_metadata(pd.read_csv(rf'{path}/{metadata_dir}'), images)

  
    ### Acne Eczema Psoriasis ###
    metadata = metadata[metadata['Diagnosis'].isin(['psoriasis', 'acne', 'eczema'])]

    ### Eczema Psoriasis ###
    #metadata = metadata[metadata['Diagnosis'].isin(['psoriasis', 'eczema'])]

    ### Cancer vs Non-Cancer ###
    #benign_conditions = ['benign', 'melanocytic nevus', 'cyst']
    #malignant_conditions = ['malignant', 'melanoma', 'bcc', 'scc']

    #metadata = metadata[metadata['Diagnosis'].isin(benign_conditions + malignant_conditions)]
    #metadata['Diagnosis'] = metadata['Diagnosis'].apply(
    #    lambda x: 'Benign' if x in benign_conditions else 'Malignant'
    #)


    if len(metadata) == 0:
            print(f"Warning: Dataset {path_folder} has no samples for the specified diagnoses. Skipping...")
            return None, None, None, None

    try:
        metadata_train, metadata_test = train_test_split(
            metadata,
            test_size=0.3,
            stratify=metadata[stratification_strategy],  
            random_state=seed
        )
    except ValueError as e:
        metadata_train, metadata_test = train_test_split(
            metadata,
            test_size=0.3,
            shuffle=True,
            random_state=seed
        )

    try:
        metadata_val, metadata_test = train_test_split(
            metadata_test,
            test_size=0.4,
            stratify=metadata_test[stratification_strategy],
            random_state=seed
        )
    except ValueError as e:
        metadata_val, metadata_test = train_test_split(
            metadata_test,
            test_size=0.4,
            shuffle=True,
            random_state=seed
        )

    return metadata_train, metadata_test, metadata_val, images