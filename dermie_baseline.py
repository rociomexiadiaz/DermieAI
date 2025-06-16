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
from xai import *

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/dermie_experiment_{experiment_timestamp}.txt"

def save_experiment_log(data, file_path=log_file):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def save_plot_and_return_path(fig, filename_base):
    filename = f"Logs/{filename_base}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

experiment_data = {}
experiment_data['Python Filename'] = os.path.basename(__file__)


### LOAD AND CLEAN METADATA ###

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(project_dir, r'Data/dermie_data')
images = rf'{path}/master_data_june_7_2025'
metadata = clean_metadata(pd.read_csv(rf'{path}/master_data_june_7_2025.csv'), images)


### VISUALISE DATA ###

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

    return fig

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

Dermie_full = DermieDataset(metadata, images, transform=transformations_val_test) 

visualise(Dermie_full)



### CREATE DATASETS + DATALOADERS ###

metadata = metadata[metadata['Diagnosis'].isin(['psoriasis', 'melanoma', 'acne', 'melanocytic nevus', 'eczema', 'scc', 'bcc', 'urticaria'])]
stratification_strategy = 'Diagnosis'  # 'stratify_col' -> Ensure all conditions and skin tones are in both train and test

metadata_train, metadata_test = train_test_split(
    metadata,
    test_size=0.3,
    stratify=metadata[stratification_strategy],  
    random_state=42
)

metadata_val, test_df = train_test_split(
    metadata_test,
    test_size=0.4,
    stratify=metadata_test[stratification_strategy],  # stratify=metadata['stratify_col'] -> Ensure all conditions and skin tones are in both val and test
    random_state=42
)

Dermie_train = DermieDataset(metadata_train, images, transform=transformations) 
Dermie_val = DermieDataset(metadata_val, images, transform=transformations_val_test, diagnostic_encoder=Dermie_train.diagnose_encoder)
Dermie_test = DermieDataset(metadata_test, images, transform=transformations_val_test, diagnostic_encoder=Dermie_train.diagnose_encoder)

fig_train = visualise(Dermie_train)
fig_test = visualise(Dermie_test)

fig_train_path = save_plot_and_return_path(fig_train, 'Train_dataset')
fig_test_path = save_plot_and_return_path(fig_test, 'Test_dataset')

conditions_mapping = Dermie_train.diagnose_encoder.categories_[0]

balancer_strategy = 'diagnostic' # or 'both'
batch_size = 64

train_sampler = BalanceSampler(Dermie_train, choice=balancer_strategy)

pad_train_dataloader = torch.utils.data.DataLoader(
    Dermie_train,
    batch_size=batch_size,
    num_workers=0,
    sampler=train_sampler
)
pad_val_dataloader = torch.utils.data.DataLoader(
    Dermie_val,
    batch_size=batch_size,
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

model = models.resnet152(weights='IMAGENET1K_V1')

num_classes = 8
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, num_classes),
)


for name, param in model.named_parameters():
    #param.requires_grad = True
    if 'fc' not in name:
        #continue
        param.requires_grad = False


### MODEL TRAINING AND TESTING ###

lr = 0.001
num_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()

model = train_model(
    model,
    pad_train_dataloader,
    pad_val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=num_epochs,
    run_folder='Dermie_runs'
)

metrics = test_model(
    model,
    pad_test_dataloader,
    device,
    top_k_accuracy(3), top_k_sensitivity(3), stratified_k_accuracy(3), stratified_k_sensitivity(3), missclassified_samples()
)   

summary = summarise_metrics(metrics, conditions_mapping)
experiment_data['Metrics'] = '\n'.join(summary)

### MODEL EXPLANATION ###

#model_gradCAM = UniversalGrad(model, 'layer4.2.conv3')
#model_gradCAM.eval()
#heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, pad_test_dataloader, device)
#fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
#grad_cam_path = save_plot_and_return_path(fig, 'gradCAM')
#experiment_data['GradCAM Plot Path'] = grad_cam_path


### SAVE RESULTS ###

experiment_data['Dataset Path'] = path 
experiment_data['Stratification Technique'] = stratification_strategy 
experiment_data['Sampler Choice'] = balancer_strategy 
experiment_data['Batch Size'] = batch_size 
experiment_data['Model'] = 'ResNet50' 
experiment_data['Learning Rate'] = lr 
experiment_data['Optimizer'] = 'Adam' 
experiment_data['Criterion'] = 'BCEWithLogitsLoss' 
experiment_data['Train Dataset Visualisation'] = fig_train_path 
experiment_data['Test Dataset Visualisation'] = fig_test_path 
save_experiment_log(experiment_data, file_path=log_file)

