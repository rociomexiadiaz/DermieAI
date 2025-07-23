from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from metricsFunctions2 import *
from Baseline import *
from TestFunction import *
import matplotlib.pyplot as plt
from xai import *
import copy

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)

def save_experiment_log(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def save_plot_and_return_path(fig, filename_base, dataset_combo):
    filename = f"Logs/{filename_base}_{dataset_combo}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


### LOAD ALL DATASETS ###

stratification_strategy = 'Diagnosis'  # 'stratify_col' -> Ensure all conditions and skin tones are in both train and test

print("Loading all datasets...")

# Load Dermie
dermie_metadata_train, dermie_metadata_test, dermie_metadata_val, images_dermie = load_dataset(
    project_dir=project_dir,
    path_folder=r'Data/dermie_data', 
    images_dir='master_data_june_7_2025.zip',
    metadata_dir='master_data_june_7_2025.csv',
    stratification_strategy=stratification_strategy
)

# Load PADUFES
pad_metadata_train, pad_metadata_test, pad_metadata_val, images_pad = load_dataset(
    project_dir=project_dir,
    path_folder=r'Data/padufes', 
    images_dir='padufes_images.zip',
    metadata_dir='padufes_metadata_clean.csv',
    stratification_strategy=stratification_strategy
)

# Load SCIN
scin_metadata_train, scin_metadata_test, scin_metadata_val, images_scin = load_dataset(
    project_dir=project_dir,
    path_folder=r'Data/scin', 
    images_dir='scin_images.zip',
    metadata_dir='scin_metadata_clean.csv',
    stratification_strategy=stratification_strategy
)

# Load Fitzpatrick17k (TABE)
fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(
    project_dir=project_dir,
    path_folder=r'Data/fitz17k', 
    images_dir='fitzpatrick17k_images.zip',
    metadata_dir='fitzpatrick17k_metadata_clean.csv',
    stratification_strategy=stratification_strategy
)

# Load India
india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(
    project_dir=project_dir,
    path_folder=r'Data/india_data', 
    images_dir='india_images.zip',
    metadata_dir='india_metadata_final.csv',
    stratification_strategy=stratification_strategy
)

print("All datasets loaded successfully!")


### DEFINE DATASET COMBINATIONS ###

dataset_combinations = {
    'All': {
        'train': [dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train],
        'val': [dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val],
        'test': [dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_test],
        'images': [images_dermie, images_pad, images_scin, images_fitz17, images_india],
        'description': 'Dermie + PADUFES + SCIN + Fitzpatrick17k + India'
    },
    'Minus_Dermie': {
        'train': [pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train],
        'val': [pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val],
        'test': [pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_test],
        'images': [images_pad, images_scin, images_fitz17, images_india],
        'description': 'PADUFES + SCIN + Fitzpatrick17k + India'
    },
    'Minus_PADUFES': {
        'train': [dermie_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train],
        'val': [dermie_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val],
        'test': [dermie_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_test],
        'images': [images_dermie, images_scin, images_fitz17, images_india],
        'description': 'Dermie + SCIN + Fitzpatrick17k + India'
    },
    'Minus_Fitz': {
        'train': [dermie_metadata_train, pad_metadata_train, scin_metadata_train, india_metadata_train],
        'val': [dermie_metadata_val, pad_metadata_val, scin_metadata_val, india_metadata_val],
        'test': [dermie_metadata_test, pad_metadata_test, scin_metadata_test, india_metadata_test],
        'images': [images_dermie, images_pad, images_scin, images_india],
        'description': 'Dermie + PADUFES + SCIN + India'
    },
    'Minus_SCIN': {
        'train': [dermie_metadata_train, pad_metadata_train, fitz17_metadata_train, india_metadata_train],
        'val': [dermie_metadata_val, pad_metadata_val, fitz17_metadata_val, india_metadata_val],
        'test': [dermie_metadata_test, pad_metadata_test, fitz17_metadata_test, india_metadata_test],
        'images': [images_dermie, images_pad, images_fitz17, images_india],
        'description': 'Dermie + PADUFES + Fitzpatrick17k + India'
    },
    'Minus_India': {
        'train': [dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train],
        'val': [dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val],
        'test': [dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test],
        'images': [images_dermie, images_pad, images_scin, images_fitz17],
        'description': 'Dermie + PADUFES + SCIN + Fitzpatrick17k'
    }
}


### TRANSFORMATIONS ###

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


### TRAINING PARAMETERS ###

balancer_strategy = 'diagnostic'  # or 'both'
batch_size = 64
lr = 0.001
num_epochs = 10
num_classes = 8


### RUN EXPERIMENTS FOR ALL DATASET COMBINATIONS ###

for combo_name, combo_data in dataset_combinations.items():
    print(f"\n{'='*50}")
    print(f"Running experiment: {combo_name}")
    print(f"Datasets: {combo_data['description']}")
    print(f"{'='*50}")
    
    # Reset random seeds for reproducibility
    torch.manual_seed(0)
    
    # Create experiment data dictionary
    experiment_data = {}
    experiment_data['Python Filename'] = os.path.basename(__file__)
    experiment_data['Dataset Combination'] = combo_name
    experiment_data['Datasets'] = combo_data['description']
    
    # Create datasets and dataloaders for this combination
    print("Creating datasets...")
    train_set = MultipleDatasets(
        combo_data['train'], 
        combo_data['images'], 
        transform=transformations
    ) 
    val_set = MultipleDatasets(
        combo_data['val'], 
        combo_data['images'], 
        transform=transformations_val_test, 
        diagnostic_encoder=train_set.diagnose_encoder
    )
    test_set = MultipleDatasets(
        combo_data['test'], 
        combo_data['images'], 
        transform=transformations_val_test, 
        diagnostic_encoder=train_set.diagnose_encoder
    )
    
    # Visualize datasets
    fig_train = visualise(train_set)
    fig_test = visualise(test_set)
    
    fig_train_path = save_plot_and_return_path(fig_train, 'Train_dataset', combo_name)
    fig_test_path = save_plot_and_return_path(fig_test, 'Test_dataset', combo_name)
    
    conditions_mapping = train_set.diagnose_encoder.categories_[0]
    
    # Create data loaders
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
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Initialize model
    print("Initializing model...")
    model = models.resnet152(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, num_classes),
    )
    
    # Freeze layers except layer4 and fc
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2], gamma=0.1)
    
    # Train model
    print("Training model...")
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
    loss_path = save_plot_and_return_path(fig, 'losses', combo_name)
    
    # Test model
    print("Testing model...")
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
    
     
    # Model explanation (GradCAM)
    print("Generating GradCAM visualizations...")
    try:
        model_gradCAM = UniversalGrad(model, 'layer4.2.conv3')
        model_gradCAM.eval()
        heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
        fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
        grad_cam_path = save_plot_and_return_path(fig, 'gradCAM', combo_name)
        experiment_data['GradCAM Plot Path'] = grad_cam_path
    except Exception as e:
        print(f"GradCAM failed for {combo_name}: {str(e)}")
        experiment_data['GradCAM Plot Path'] = f"Failed: {str(e)}"
    
    # Save experiment results
    experiment_data['Train Dataset Visualisation'] = fig_train_path 
    experiment_data['Test Dataset Visualisation'] = fig_test_path 
    experiment_data['Loss Plot Path'] = loss_path
    
    # Save experiment log
    log_file = f"Logs/dermie_experiment_{combo_name}_{experiment_timestamp}.txt"
    save_experiment_log(experiment_data, file_path=log_file)
    
    print(f"Experiment {combo_name} completed!")
    
    # Clear GPU memory
    del model, train_dataloader, val_dataloader, test_dataloader
    del train_set, val_set, test_set
    torch.cuda.empty_cache()

print(f"\n{'='*50}")
print("ALL EXPERIMENTS COMPLETED!")
print(f"Results saved with timestamp: {experiment_timestamp}")
print(f"{'='*50}")