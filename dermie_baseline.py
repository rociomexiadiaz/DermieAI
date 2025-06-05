import pandas as pd
from dermie_dataset import *
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torchvision import models
from metricsFunctions import *
from TrainValFunctions import *
from TestFunction import *
import matplotlib.pyplot as plt

### SEEDS AND DEVICE ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### LOAD AND CLEAN METADATA ###

path = 'dermie_images'
metadata = clean_metadata(pd.read_csv(f'{path}/Labelled_images_for_training_700_15_disease.csv'))
images = f'{path}/Labelled_images_for_training_700'


### REMOVE FROM METADADA IMAGES NOT LOADED ###

images_loaded = os.listdir(images)
metadata = metadata[metadata['Image Name'].isin(images_loaded)]


### NARROW CONDITIONS ###

metadata = metadata[metadata['Diagnosis'].isin(['eczema', 'acne', 'psoriasis', 'urticaria'])]

### CREATE DATASETS ###

metadata_train, metadata_test = train_test_split(
    metadata,
    test_size=0.3,
    stratify=metadata['Diagnosis'],  # stratify=metadata['stratify_col'] -> Ensure all conditions and skin tones are in both train and test
    random_state=42
)

metadata_val, test_df = train_test_split(
    metadata_test,
    test_size=0.4,
    stratify=metadata_test['Diagnosis'],  # stratify=metadata['stratify_col'] -> Ensure all conditions and skin tones are in both val and test
    random_state=42
)

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomAffine(degrees=10, shear= (-10,10,-10,10)),
])

transformations_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

Dermie_train = DermieDataset(metadata_train, images, transform=transformations) 
Dermie_val = DermieDataset(metadata_val, images, transform=transformations_val_test, diagnostic_encoder=Dermie_train.diagnose_encoder)
Dermie_test = DermieDataset(metadata_test, images, transform=transformations_val_test, diagnostic_encoder=Dermie_train.diagnose_encoder)

conditions_mapping = Dermie_train.diagnose_encoder.categories_[0]

train_sampler = BalanceSampler(Dermie_train, choice='diagnostic')

pad_train_dataloader = torch.utils.data.DataLoader(
    Dermie_train,
    batch_size=64,
    num_workers=0,
    sampler=train_sampler
)
pad_val_dataloader = torch.utils.data.DataLoader(
    Dermie_val,
    batch_size=64,
    shuffle=False,
    num_workers=0
)
pad_test_dataloader = torch.utils.data.DataLoader(
    Dermie_val,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

### MODEL LOADING ###

model = models.resnet50(weights='IMAGENET1K_V1')

num_classes = 4
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, num_classes),
    #torch.nn.Softmax(dim=1)  nn.CrossEntropyLoss already applies softmax
)

for name, param in model.named_parameters():
    #param.requires_grad = True
    if 'fc' not in name:
        #continue
        param.requires_grad = False


### MODEL TRAINING AND TESTING ###

model = train_model(
    model,
    pad_train_dataloader,
    pad_val_dataloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    criterion=torch.nn.BCEWithLogitsLoss(),
    device=device,
    num_epochs=5,
    run_folder='Dermie_runs'
)

metrics = test_model(
    model,
    pad_test_dataloader,
    device,
    top_k_accuracy(3), top_k_sensitivity(3), stratified_k_accuracy(3), stratified_k_sensitivity(3), missclassified_samples()
)   

summarise_metrics(metrics, conditions_mapping)


### Check individual dataset distributions

def visualise(dataset: DermieDataset):

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

visualise(Dermie_train)
visualise(Dermie_test)

