from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from TABE import *
from TestFunction import *
from metricsFunctions2 import *
import matplotlib.pyplot as plt
from xai import *

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/dermie_experiment_{experiment_timestamp}.txt"

with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"Python Filename: {os.path.basename(__file__)}\n")
    f.write(f"Experiment Timestamp: {experiment_timestamp}\n\n")

def append_experiment_log(data, file_path=log_file):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"=== Fold: {data['Fold']} ===\n")
        for key, value in data.items():
            if key != 'Fold':
                f.write(f"{key}: {value}\n")
        f.write("\n")

def save_plot_and_return_path(fig, filename_base):
    filename = f"Logs/{filename_base}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename


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

datasets = [
    ('dermie', dermie_metadata_train, dermie_metadata_val, dermie_metadata_test, images_dermie),
    ('padufes', pad_metadata_train, pad_metadata_val, pad_metadata_test, images_pad),
    ('scin', scin_metadata_train, scin_metadata_val, scin_metadata_test, images_scin),
    ('fitz17k', fitz17_metadata_train, fitz17_metadata_val, fitz17_metadata_test, images_fitz17),
    ('india', india_metadata_train, india_metadata_val, india_metadata_test, images_india),
]

for i, (test_name, test_train, test_val, test_test, test_images) in enumerate(datasets):
    print(f'\n=== Running fold with {test_name} as test set ===')

    experiment_data = {}
    experiment_data['Fold'] = test_name

    # Test set - all splits of test dataset
    test_metadata = pd.concat([test_train, test_val, test_test], ignore_index=True)
    test_set = MultipleDatasets([test_metadata], [test_images], transform=transformations_val_test)

    # Train and Val
    train_metadatas, train_images = [], []
    val_metadatas, val_images = [], []

    for j, (name, train, val, test, images) in enumerate(datasets):
        if i == j:
            continue  # skip test dataset

        train_metadatas.append(train)
        train_images.append(images)

        val_combined = pd.concat([val, test], ignore_index=True)
        val_metadatas.append(val_combined)
        val_images.append(images)

    train_set = MultipleDatasets(train_metadatas, train_images, transform=transformations)
    val_set = MultipleDatasets(val_metadatas, val_images, transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)


    fig_train = visualise(train_set)
    fig_test = visualise(test_set)

    fig_train_path = save_plot_and_return_path(fig_train, f'{test_name}_Train_dataset')
    fig_test_path = save_plot_and_return_path(fig_test, f'{test_name}_Test_dataset')

    experiment_data['Train Dataset Visualisation'] = fig_train_path
    experiment_data['Test Dataset Visualisation'] = fig_test_path

    conditions_mapping = train_set.diagnose_encoder.categories_[0]
    num_conditions = len(conditions_mapping)

    balancer_strategy = 'diagnostic' # or 'both'
    batch_size = 64

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

    model_encoder = FeatureExtractor(enet=models.resnet152(weights="IMAGENET1K_V2"))
    model_classifier = ClassificationHead(out_dim=num_conditions, in_ch=model_encoder.in_ch)
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

    loss_path = save_plot_and_return_path(fig, f'{test_name}_losses')
    experiment_data['Loss Curve'] = loss_path

    model = nn.Sequential(
        model_encoder,
        model_classifier,
        model_classifier.activation)

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
    experiment_data['Metrics'] = '\n'.join(summary)


    ### MODEL EXPLANATION ###

    model_gradCAM = UniversalGrad(model, '0.enet.layer4.2.conv3')
    model_gradCAM.eval()
    heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
    
    fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
    grad_cam_path = save_plot_and_return_path(fig, f'{test_name}_gradCAM')
    experiment_data['GradCAM Plot Path'] = grad_cam_path

    ### SAVE RESULTS ###

    append_experiment_log(experiment_data, file_path=log_file)   