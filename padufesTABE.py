import pandas as pd
import torch
from padufes_dataset import PADUFESDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from TABE import *
from TestFunction import *
from metricsFunctions import *
from xai import *

### SETTING SEEDS AND DEVICE ###

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
    num_workers=0,
    sampler=None
)
pad_val_dataloader = torch.utils.data.DataLoader(
    PAD_val,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    sampler=None
)
pad_test_dataloader = torch.utils.data.DataLoader(
    PAD_test,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    sampler=None
)

### MODEL LOADING ###

model_encoder = FeatureExtractor()
model_classifier = ClassificationHead(out_dim=6, in_ch=model_encoder.in_ch)
model_aux = AuxiliaryHead(num_aux=6, in_ch=model_encoder.in_ch)

optimizer = torch.optim.SGD(list(model_encoder.parameters()) + list(model_classifier.parameters()) + list(model_aux.parameters()),
                            lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer_confusion = torch.optim.SGD(model_encoder.parameters(), lr=0.001, momentum=0.9)  # Defining confusion optimiser (boosted encoder optimiser)
optimizer_aux = torch.optim.SGD(model_aux.parameters(), lr=0.001, momentum=0.9)  # defining auxiliary classification optimiser

criterion = nn.CrossEntropyLoss()
criterion_aux = nn.CrossEntropyLoss()

alpha = 0.5  
GRL = True  

model = nn.Sequential(
    model_encoder,
    model_classifier,
    model_classifier.activation)

### MODEL TRAINING AND TESTING ###

model_encoder, model_classifier, model_aux = train_model(
    model_encoder=model_encoder, 
    model_classifier=model_classifier, 
    model_aux=model_aux,
    train_loader=pad_train_dataloader, 
    val_loader=pad_val_dataloader,
    num_epochs=1, 
    optimizer=optimizer, 
    optimizer_aux=optimizer_aux, 
    optimizer_confusion=optimizer_confusion,
    criterion=criterion, 
    criterion_aux=criterion_aux, 
    device=device, 
    alpha=alpha, 
    GRL=GRL
)

model = nn.Sequential(
    model_encoder,
    model_classifier,
    model_classifier.activation)

metrics = test_model(model, pad_test_dataloader, device, top_k_accuracy(3), top_k_sensitivity(3), stratified_k_accuracy(3), stratified_k_sensitivity(3), missclassified_samples())   

summarise_metrics(metrics, conditions_mapping)


### MODEL EXPLANATION ###

model_gradCAM = UniversalGrad(model, '0.enet.layer4.2.conv3')
model_gradCAM.eval()
heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, pad_test_dataloader, device)
visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)

