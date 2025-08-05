from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from TABE import *
from TestFunction import *
from metricsFunctions import *
import datetime
import matplotlib.pyplot as plt
from xai import *

clip_fe = False

### SEEDS, DEVICE AND LOG FILE  ###

torch.cuda.empty_cache()
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
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


### LOAD DATA ###

stratification_strategy = 'Diagnosis'  # 'stratify_col' -> Ensure all conditions and skin tones are in both train and test

dermie_metadata_train, dermie_metadata_test, dermie_metadata_val, images_dermie = load_dataset(project_dir=project_dir,
                                                                                               path_folder=r'Data/dermie_data', 
                                                                                               images_dir='master_data_june_7_2025.zip',
                                                                                               metadata_dir='master_data_june_7_2025.csv',
                                                                                               stratification_strategy=stratification_strategy)

pad_metadata_train, pad_metadata_test, pad_metadata_val, images_pad = load_dataset(project_dir=project_dir,
                                                                                   path_folder=r'Data/padufes', 
                                                                                   images_dir='padufes_images.zip',
                                                                                   metadata_dir='padufes_metadata_new.csv',
                                                                                   stratification_strategy=stratification_strategy)

scin_metadata_train, scin_metadata_test, scin_metadata_val, images_scin = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/scin', 
                                                                                       images_dir='scin_images.zip',
                                                                                       metadata_dir='scin_metadata_new.csv',
                                                                                       stratification_strategy=stratification_strategy)

fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/fitz17k', 
                                                                                       images_dir='fitzpatrick17k_images.zip',
                                                                                       metadata_dir='fitzpatrick17k_metadata_new.csv',
                                                                                       stratification_strategy=stratification_strategy)

india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/india_data', 
                                                                                       images_dir='india_images.zip',
                                                                                       metadata_dir='india_metadata_final.csv',
                                                                                       stratification_strategy=stratification_strategy)

experiment_data['Datasets'] = 'Dermie + Padufes + SCIN + Fitzpatrick17k + India'


### CREATE DATASETS AND DATALOADERS ###

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

#train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=transformations) 
#val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)
#test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)

train_set = MultipleDatasets([fitz17_metadata_train], [images_fitz17], transform=transformations) 
val_set = MultipleDatasets([fitz17_metadata_val], [images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)
test_set = MultipleDatasets([fitz17_metadata_test], [images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)

# CLIP
if clip_fe:
    train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=transformations, clip=True, apply_augment=True) 
    val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=None, diagnostic_encoder=train_set.diagnose_encoder, clip=True, apply_augment=False)
    test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=None, diagnostic_encoder=train_set.diagnose_encoder, clip=True, apply_augment=False)

fig_train = visualise(train_set)
fig_test = visualise(test_set)

fig_train_path = save_plot_and_return_path(fig_train, 'Train_dataset')
fig_test_path = save_plot_and_return_path(fig_test, 'Test_dataset')

conditions_mapping = train_set.diagnose_encoder.categories_[0]
num_conditions = len(conditions_mapping)

balancer_strategy = 'diagnostic' # or 'both'
batch_size = 32

train_sampler = BalanceSampler(train_set, choice=balancer_strategy)

train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=0,
    sampler=train_sampler
)
val_dataloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


### MODEL LOADING ###

class FC(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim) 
        self.in_ch = output_dim  

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  
        return self.fc(x) 
  

# CLIP
if clip_fe:
    model_encoder = FC()
else:
    model_encoder = FeatureExtractor(enet=models.resnet152(weights="IMAGENET1K_V2"))

model_classifier = ClassificationHead(out_dim=num_conditions, in_ch=model_encoder.in_ch)
model_aux = AuxiliaryHead(num_aux=6, in_ch=model_encoder.in_ch)

optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_classifier.parameters()) + list(model_aux.parameters()),
                            lr=0.001)
optimizer_confusion = torch.optim.Adam(model_encoder.parameters(), lr=0.001)  
optimizer_aux = torch.optim.Adam(model_aux.parameters(), lr=0.001) 

criterion = nn.CrossEntropyLoss()
criterion_aux = nn.CrossEntropyLoss()

alpha = 0.8
GRL = True  

model = nn.Sequential(
    model_encoder,
    model_classifier,
    model_classifier.activation)

### MODEL TRAINING AND TESTING ###

model_encoder, model_classifier, model_aux, fig = train_model(
    model_encoder=model_encoder, 
    model_classifier=model_classifier, 
    model_aux=model_aux,
    train_loader=train_dataloader, 
    val_loader=val_dataloader,
    num_epochs=15, 
    optimizer=optimizer, 
    optimizer_aux=optimizer_aux, 
    optimizer_confusion=optimizer_confusion,
    criterion=criterion, 
    criterion_aux=criterion_aux, 
    device=device, 
    alpha=alpha, 
    GRL=GRL
)

loss_path = save_plot_and_return_path(fig, 'losses')

model = nn.Sequential(
    model_encoder,
    model_classifier,
    model_classifier.activation)

if num_conditions > 5:
    metrics = test_model(
        model,
        test_dataloader,
        device,
        multi_k_accuracy([1, 3, 5]),
        multi_k_sensitivity([1, 3, 5]),
        stratified_multi_k_accuracy([1, 3, 5]),
        stratified_multi_k_sensitivity([1, 3, 5]),
        enhanced_misclassified_samples(),
        f1_score_metric(),
        stratified_f1_score(),
        balanced_accuracy(),
        stratified_balanced_accuracy()
    )

    # Summarize metrics
    summary = summarise_enhanced_metrics(metrics, conditions_mapping, k_values=[1, 3, 5])
    experiment_data['Metrics'] = '\n'.join(summary)

else:
    metrics = test_model(
    model,
    test_dataloader,
    device,
    multi_k_accuracy([1]),
    multi_k_sensitivity([1]),
    stratified_multi_k_accuracy([1]),
    stratified_multi_k_sensitivity([1]),
    enhanced_misclassified_samples(),
    f1_score_metric(),
    stratified_f1_score(),
    balanced_accuracy(),
    stratified_balanced_accuracy()
)
    
    # Summarize metrics
    summary = summarise_enhanced_metrics(metrics, conditions_mapping, k_values=[1])
    experiment_data['Metrics'] = '\n'.join(summary)


### MODEL EXPLANATION ###

if not clip_fe:
    model_gradCAM = UniversalGrad(model, '0.enet.layer4.2.conv3')
    model_gradCAM.eval()
    heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
    fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
    grad_cam_path = save_plot_and_return_path(fig, 'gradCAM')
    experiment_data['GradCAM Plot Path'] = grad_cam_path


### SAVE RESULTS ###

experiment_data['Train Dataset Visualisation'] = fig_train_path 
experiment_data['Test Dataset Visualisation'] = fig_test_path 
save_experiment_log(experiment_data, file_path=log_file)

