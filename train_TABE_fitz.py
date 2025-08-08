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
import pandas as pd
from sklearn.model_selection import train_test_split

clip_fe = False

### SEEDS, DEVICE AND LOG FILE  ###

torch.cuda.empty_cache()
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/dermie_experiment_fst_{experiment_timestamp}.txt"

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


### LOAD AND MERGE DATA ###


stratification_strategy = 'Diagnosis'  

fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/fitz17k', 
                                                                                       images_dir='fitzpatrick17k_images.zip',
                                                                                       metadata_dir='fitzpatrick17k_metadata_new.csv',
                                                                                       stratification_strategy=stratification_strategy)

all_metadata = pd.concat([fitz17_metadata_train, fitz17_metadata_val, fitz17_metadata_test], ignore_index=True)

train_fst_metadata = all_metadata[all_metadata['Fitzpatrick'].isin(['I', 'II'])].reset_index(drop=True)

test_fst_34_metadata = all_metadata[all_metadata['Fitzpatrick'].isin(['III', 'IV'])].reset_index(drop=True)

test_fst_56_metadata = all_metadata[all_metadata['Fitzpatrick'].isin(['V', 'VI'])].reset_index(drop=True)

train_metadata_final, val_metadata_final = train_test_split(
    train_fst_metadata, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_fst_metadata['Diagnosis']
)

print(f"Training set size (FST I,II): {len(train_metadata_final)}")
print(f"Validation set size (FST I,II): {len(val_metadata_final)}")
print(f"Test set size (FST III,IV): {len(test_fst_34_metadata)}")
print(f"Test set size (FST V,VI): {len(test_fst_56_metadata)}")

experiment_data['Datasets'] = 'Fitzpatrick17k - FST Split'
experiment_data['Train FST'] = 'I, II'
experiment_data['Test Group 1 FST'] = 'III, IV'
experiment_data['Test Group 2 FST'] = 'V, VI'


### CREATE DATASETS AND DATALOADERS ###

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.RandomAffine(degrees=30, shear= (-10,10,-10,10)), # 30 instead of 10
    transforms.ColorJitter(brightness=0.1), # NEW
    transforms.RandomHorizontalFlip(p=0.5), # NEW
    transforms.RandomVerticalFlip(p=0.2), # NEW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transformations_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_set = MultipleDatasets([train_metadata_final], [images_fitz17], transform=transformations) 
val_set = MultipleDatasets([val_metadata_final], [images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)
test_set_34 = MultipleDatasets([test_fst_34_metadata], [images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)
test_set_56 = MultipleDatasets([test_fst_56_metadata], [images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)

fig_train = visualise(train_set)
fig_test_34 = visualise(test_set_34)
fig_test_56 = visualise(test_set_56)

fig_train_path = save_plot_and_return_path(fig_train, 'Train_dataset_FST_I_II')
fig_test_34_path = save_plot_and_return_path(fig_test_34, 'Test_dataset_FST_III_IV')
fig_test_56_path = save_plot_and_return_path(fig_test_56, 'Test_dataset_FST_V_VI')

conditions_mapping = train_set.diagnose_encoder.categories_[0]
num_conditions = len(conditions_mapping)

balancer_strategy = 'diagnostic' # or 'both'
batch_size = 16 #32

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
test_dataloader_34 = torch.utils.data.DataLoader(
    test_set_34,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_dataloader_56 = torch.utils.data.DataLoader(
    test_set_56,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


### MODEL LOADING ###

#model_encoder = FeatureExtractor(enet=models.resnet152(weights="IMAGENET1K_V2"))
#model_classifier = ClassificationHead(out_dim=num_conditions, in_ch=model_encoder.in_ch)
#model_aux = AuxiliaryHead(num_aux=6, in_ch=model_encoder.in_ch)

#optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_classifier.parameters()) + list(model_aux.parameters()),
                            #lr=0.001) 
#optimizer_confusion = torch.optim.Adam(model_encoder.parameters(), lr=0.001)  
#optimizer_aux = torch.optim.Adam(model_aux.parameters(), lr=0.001) 

#criterion = nn.CrossEntropyLoss()
#criterion_aux = nn.CrossEntropyLoss()

#alpha = 0.1
#GRL = True  

#model = nn.Sequential(
    #model_encoder,
    #model_classifier,
    #model_classifier.activation)

### MODEL TRAINING AND TESTING ###

#model_encoder, model_classifier, model_aux, fig = train_model(
    #model_encoder=model_encoder, 
    #model_classifier=model_classifier, 
    #model_aux=model_aux,
    #train_loader=train_dataloader, 
    #val_loader=val_dataloader,
    #num_epochs=15, 
    #optimizer=optimizer, 
    #optimizer_aux=optimizer_aux, 
    #optimizer_confusion=optimizer_confusion,
    #criterion=criterion, 
    #criterion_aux=criterion_aux, 
    #device=device, 
    #alpha=alpha, 
    #GRL=GRL
#)

#loss_path = save_plot_and_return_path(fig, 'losses')

#model = nn.Sequential(
    #model_encoder,
    #model_classifier,
    #model_classifier.activation)



model = models.resnet152(weights='IMAGENET1K_V1')

model.fc = torch.nn.Sequential(
    #torch.nn.Dropout(0.1), # NEW 
    torch.nn.Linear(model.fc.in_features, num_conditions),
    
)

for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


class FC(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim) 
        self.in_ch = input_dim

    def forward(self, x):       
        return self.fc(x) 
  

### MODEL TRAINING AND TESTING ###

lr = 0.001
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
from Baseline import *
model, fig = train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    num_epochs=num_epochs
)
loss_path = save_plot_and_return_path(fig, 'losses')

# Test on FST III,IV (Group 1)
print("Testing on FST III,IV...")
if num_conditions > 5:
    metrics_34 = test_model(
        model,
        test_dataloader_34,
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

    # Summarize metrics for FST III,IV
    summary_34 = summarise_enhanced_metrics(metrics_34, conditions_mapping, k_values=[1, 3, 5])
    experiment_data['Metrics FST III,IV'] = '\n'.join(summary_34)

else:
    metrics_34 = test_model(
        model,
        test_dataloader_34,
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
    
    # Summarize metrics for FST III,IV
    summary_34 = summarise_enhanced_metrics(metrics_34, conditions_mapping, k_values=[1])
    experiment_data['Metrics FST III,IV'] = '\n'.join(summary_34)

# Test on FST V,VI (Group 2)
print("Testing on FST V,VI...")
if num_conditions > 5:
    metrics_56 = test_model(
        model,
        test_dataloader_56,
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

    # Summarize metrics for FST V,VI
    summary_56 = summarise_enhanced_metrics(metrics_56, conditions_mapping, k_values=[1, 3, 5])
    experiment_data['Metrics FST V,VI'] = '\n'.join(summary_56)

else:
    metrics_56 = test_model(
        model,
        test_dataloader_56,
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
    
    # Summarize metrics for FST V,VI
    summary_56 = summarise_enhanced_metrics(metrics_56, conditions_mapping, k_values=[1])
    experiment_data['Metrics FST V,VI'] = '\n'.join(summary_56)


### MODEL EXPLANATION ###

# GradCAM for FST III,IV
#model_gradCAM = UniversalGrad(model, '0.enet.layer4.2.conv3')
model_gradCAM = UniversalGrad(model, 'layer4.2.conv3')

model_gradCAM.eval()
heatmaps_34, images_for_grad_cam_34, predicted_labels_34, real_labels_34 = gradCAM(model_gradCAM, test_dataloader_34, device)
fig_34 = visualize_gradcams_with_colorbars(images_for_grad_cam_34, heatmaps_34, predicted_labels_34, real_labels_34, conditions_mapping)
grad_cam_34_path = save_plot_and_return_path(fig_34, 'gradCAM_FST_III_IV')
experiment_data['GradCAM FST III,IV Path'] = grad_cam_34_path

# GradCAM for FST V,VI
heatmaps_56, images_for_grad_cam_56, predicted_labels_56, real_labels_56 = gradCAM(model_gradCAM, test_dataloader_56, device)
fig_56 = visualize_gradcams_with_colorbars(images_for_grad_cam_56, heatmaps_56, predicted_labels_56, real_labels_56, conditions_mapping)
grad_cam_56_path = save_plot_and_return_path(fig_56, 'gradCAM_FST_V_VI')
experiment_data['GradCAM FST V,VI Path'] = grad_cam_56_path


### SAVE RESULTS ###

experiment_data['Train Dataset Visualisation'] = fig_train_path 
experiment_data['Test Dataset FST III,IV Visualisation'] = fig_test_34_path 
experiment_data['Test Dataset FST V,VI Visualisation'] = fig_test_56_path
experiment_data['Loss Plot Path'] = loss_path
save_experiment_log(experiment_data, file_path=log_file)

print("Experiment completed!")
print(f"Results saved to: {log_file}")
print("\nDataset sizes:")
print(f"Training (FST I,II): {len(train_set)}")
print(f"Validation (FST I,II): {len(val_set)}")
print(f"Test Group 1 (FST III,IV): {len(test_set_34)}")
print(f"Test Group 2 (FST V,VI): {len(test_set_56)}")