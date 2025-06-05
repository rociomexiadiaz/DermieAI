import pandas as pd
import torch
from padufes_dataset import PADUFESDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import torch
from FairDisCo import *
from TestFunction import *
from metricsFunctions import *

### SETTING SEED AND DEVICE ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### DATA LOADING AND PREPROCESSING ###

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(project_dir, 'Data/PADUFES')
images = f'{path}/imgs_part_1'
metadata = pd.read_csv(f'{path}/metadata.csv')

metadata_processed = metadata[~metadata['fitspatrick'].isna()]
metadata_processed = metadata_processed[['img_id', 'fitspatrick','diagnostic']]
metadata_processed = metadata_processed[metadata_processed['img_id'].isin(os.listdir(images))]

metadata_train, metadata_test = train_test_split(metadata_processed, test_size=0.2, random_state=42)
metadata_val, metadata_test = train_test_split(metadata_test, test_size=0.2, random_state=42)


### DATASET AND DATALOADER ###

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomAffine(degrees=10, shear= (-10,10,-10,10)),
])

PAD_train = PADUFESDataset(metadata_train, images, transform=transformations) 
PAD_val = PADUFESDataset(metadata_val, images, transform=transformations, diagnostic_encoder=PAD_train.diagnose_encoder)
PAD_test = PADUFESDataset(metadata_test, images, transform=transformations, diagnostic_encoder=PAD_train.diagnose_encoder)

conditions_mapping = PAD_train.diagnose_encoder.categories_[0]

pad_train_dataloader = torch.utils.data.DataLoader(
    PAD_train,
    batch_size=64,
    shuffle=True,
    num_workers=0
)
pad_val_dataloader = torch.utils.data.DataLoader(
    PAD_val,
    batch_size=64,
    shuffle=True,
    num_workers=0
)
pad_test_dataloader = torch.utils.data.DataLoader(
    PAD_test,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

### MODEL TRAINING AND TESTING ###

model = Network([6,6])
model = train_model(model, pad_train_dataloader, pad_val_dataloader, device, num_epochs=1)

model = nn.Sequential(
    model.feature_extractor,
    model.branch_1
)

metrics = test_model(model, pad_test_dataloader, device, top_k_accuracy(3), top_k_sensitivity(3), stratified_k_accuracy(3), stratified_k_sensitivity(3), missclassified_samples()
)   

summarise_metrics(metrics, conditions_mapping)