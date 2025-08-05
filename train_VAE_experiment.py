from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from VAE import *
from TestFunction import *
from metricsFunctions import *
import matplotlib.pyplot as plt
from xai import *
import datetime

clip_fe = False

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/vae_dermie_experiment_{experiment_timestamp}.txt"

def save_experiment_log(data, file_path=log_file):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        for key, value in data.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

def save_plot_and_return_path(fig, filename_base):
    filename = f"Logs/{filename_base}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

# Initialize experiment log with header
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"VAE DERMIE COMPARISON EXPERIMENT\n")
    f.write(f"Python Filename: {os.path.basename(__file__)}\n")
    f.write(f"Experiment Timestamp: {experiment_timestamp}\n")
    f.write(f"Experiment Design: Compare VAE performance with and without Dermie dataset\n")
    f.write(f"Test Set: Same for both experiments (all datasets)\n\n")

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### LOAD DATA ###

stratification_strategy = 'Diagnosis'

print("Loading all datasets...")

dermie_metadata_train, dermie_metadata_test, dermie_metadata_val, images_dermie = load_dataset(project_dir=project_dir,
                                                                                               path_folder=r'Data/dermie_data', 
                                                                                               images_dir='master_data_june_7_2025.zip',
                                                                                               metadata_dir='master_data_june_7_2025.csv',
                                                                                               stratification_strategy=stratification_strategy)

pad_metadata_train, pad_metadata_test, pad_metadata_val, images_pad = load_dataset(project_dir=project_dir,
                                                                                   path_folder=r'Data/padufes', 
                                                                                   images_dir='padufes_images.zip',
                                                                                   metadata_dir='padufes_metadata_clean.csv',
                                                                                   stratification_strategy=stratification_strategy)

scin_metadata_train, scin_metadata_test, scin_metadata_val, images_scin = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/scin', 
                                                                                       images_dir='scin_images.zip',
                                                                                       metadata_dir='scin_metadata_clean.csv',
                                                                                       stratification_strategy=stratification_strategy)

fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/fitz17k', 
                                                                                       images_dir='fitzpatrick17k_images.zip',
                                                                                       metadata_dir='fitzpatrick17k_metadata_clean.csv',
                                                                                       stratification_strategy=stratification_strategy)

india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/india_data', 
                                                                                       images_dir='india_images.zip',
                                                                                       metadata_dir='india_metadata_final.csv',
                                                                                       stratification_strategy=stratification_strategy)

print("All datasets loaded successfully!")

### CREATE TRANSFORMATIONS ###

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomAffine(degrees=10, shear=(-10,10,-10,10)),
])

transformations_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

### CREATE DATASETS WITH AND WITHOUT DERMIE ###

# WITHOUT DERMIE (run first)
train_set_no_dermie = MultipleDatasets([pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], 
                                      [images_pad, images_scin, images_fitz17, images_india], 
                                      transform=transformations) 
val_set_no_dermie = MultipleDatasets([pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val], 
                                    [images_pad, images_scin, images_fitz17, images_india], 
                                    transform=transformations_val_test, 
                                    diagnostic_encoder=train_set_no_dermie.diagnose_encoder)

# WITH DERMIE (run second)
train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], 
                            [images_dermie, images_pad, images_scin, images_fitz17, images_india], 
                            transform=transformations) 
val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val], 
                          [images_dermie, images_pad, images_scin, images_fitz17, images_india], 
                          transform=transformations_val_test, 
                          diagnostic_encoder=train_set_no_dermie.diagnose_encoder)

# CONSISTENT TEST SET (same for both experiments)
test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_test], 
                           [images_dermie, images_pad, images_scin, images_fitz17, images_india], 
                           transform=transformations_val_test, 
                           diagnostic_encoder=train_set_no_dermie.diagnose_encoder)

print(f"Test set created with {len(test_set)} samples (consistent for both experiments)")

### EXPERIMENT COMBINATIONS ###

combinations = [
    (train_set_no_dermie, val_set_no_dermie, test_set, "NO_DERMIE"),
    (train_set, val_set, test_set, "WITH_DERMIE")
]

### TRAINING PARAMETERS ###
balancer_strategy = 'diagnostic'
batch_size = 32
resampler_alpha = 0.1
num_epochs = 20

### RUN EXPERIMENTS ###

# Store results for both experiments - NO LOGGING DURING LOOP
experiment_results = {}

for i, combination in enumerate(combinations):
    train_set_current = combination[0]
    val_set_current = combination[1]
    test_set_current = combination[2]
    experiment_name = combination[3]

    print(f"\n=== EXPERIMENT {i+1}: {experiment_name} ===")
    print(f"Training set size: {len(train_set_current)} samples")

    # Create visualizations
    fig_train = visualise(train_set_current)
    fig_test = visualise(test_set_current)

    fig_train_path = save_plot_and_return_path(fig_train, f'{experiment_name}_Train_dataset')
    fig_test_path = save_plot_and_return_path(fig_test, f'{experiment_name}_Test_dataset')

    conditions_mapping = train_set_current.diagnose_encoder.categories_[0]
    num_conditions = len(conditions_mapping)

    # Create dataloaders
    train_sampler = BalanceSampler(train_set_current, choice=balancer_strategy)

    train_dataloader = torch.utils.data.DataLoader(
        train_set_current,
        batch_size=batch_size,
        num_workers=0,
        sampler=train_sampler
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set_current,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set_current,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    ### MODEL LOADING ###
    print(f"Creating VAE model for {experiment_name}...")

    class FC(nn.Module):
        def __init__(self, input_dim=768, output_dim=256):
            super(FC, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim) 
            self.in_ch = input_dim

        def forward(self, x):
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)  
            
            return self.fc(x)  
      
    model = VAEmodel(encoder=models.resnet152(weights="IMAGENET1K_V2"), num_classes=num_conditions)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    resampler = AdaptiveResampler(alpha=resampler_alpha)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    ### MODEL TRAINING ###
    print(f"Training VAE model for {experiment_name}...")
    
    model, fig = train_VAE(model, train_dataloader, val_dataloader, optimizer, scheduler, resampler, 
                          device=device, num_epochs=num_epochs, use_clip=clip_fe)
    loss_path = save_plot_and_return_path(fig, f'{experiment_name}_losses')

    # Create final model for testing
    model = nn.Sequential(
        model.encoder,
        model.classifier,
    )

    ### MODEL TESTING ###
    print(f"Testing {experiment_name} model...")
    
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

        summary = summarise_enhanced_metrics(metrics, conditions_mapping, k_values=[1, 3, 5])

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
        
        summary = summarise_enhanced_metrics(metrics, conditions_mapping, k_values=[1])

    ### MODEL EXPLANATION ###

    grad_cam_path = None
    if not clip_fe:
        model_gradCAM = UniversalGrad(model, '0.layer4.2.conv3')
        model_gradCAM.eval()
        heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
        fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
        grad_cam_path = save_plot_and_return_path(fig, f'{experiment_name}_gradCAM')

    ### STORE RESULTS (NO LOGGING YET) ###
    training_datasets = 'Dermie + Padufes + SCIN + Fitzpatrick17k + India' if experiment_name == "WITH_DERMIE" else 'Padufes + SCIN + Fitzpatrick17k + India'
    
    experiment_results[experiment_name] = {
        'Experiment': experiment_name,
        'Training Set Size': len(train_set_current),
        'Training Datasets': training_datasets,
        'Test Set Size': len(test_set_current),
        'Metrics': '\n'.join(summary),
        'Train Dataset Visualisation': fig_train_path,
        'Test Dataset Visualisation': fig_test_path,
        'Loss Plot Path': loss_path,
        'Resampler Alpha': resampler_alpha,
        'Epochs': num_epochs,
        'Batch Size': batch_size,
        'Balancer Strategy': balancer_strategy,
        'Use CLIP': clip_fe
    }
    
    if grad_cam_path:
        experiment_results[experiment_name]['GradCAM Plot Path'] = grad_cam_path

# WRITE ALL RESULTS TO SINGLE FILE AFTER BOTH EXPERIMENTS
for experiment_name, data in experiment_results.items():
    save_experiment_log(data)

print(f"\n=== VAE EXPERIMENT COMPLETED ===")
print(f"Results logged to: {log_file}")

with open(log_file, 'a', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("VAE EXPERIMENT SUMMARY\n")
    f.write("="*80 + "\n")
    f.write("Two VAE experiments completed:\n")
    f.write("1. NO_DERMIE: Training without Dermie dataset\n")
    f.write("2. WITH_DERMIE: Training with Dermie dataset included\n")
    f.write("Both tested on the same test set for fair comparison.\n")
    f.write(f"All results saved to single log file: {log_file}\n")