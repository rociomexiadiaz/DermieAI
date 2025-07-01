from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from toneClassifier import *
import matplotlib.pyplot as plt
from xai import *
import datetime

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
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


### LOAD DATA ###

stratification_strategy = 'Fitzpatrick' 

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

experiment_data['Datasets'] = 'Dermie + Padufes + SCIN + Fitzpatrick17k'


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

train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17], transform=transformations) 
val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)
test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test], [images_dermie, images_pad, images_scin, images_fitz17], transform=transformations_val_test, diagnostic_encoder=train_set.diagnose_encoder)

fig_train = visualise(train_set)
fig_test = visualise(test_set)

fig_train_path = save_plot_and_return_path(fig_train, 'Train_dataset')
fig_test_path = save_plot_and_return_path(fig_test, 'Test_dataset')

conditions_mapping = train_set.diagnose_encoder.categories_[0]

batch_size = 64

train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
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

model = models.resnet152(weights='IMAGENET1K_V1')

num_classes = 6
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, num_classes)
)

for name, param in model.named_parameters():
    #param.requires_grad = True
    if 'fc' not in name:
        #continue
        param.requires_grad = False


### MODEL TRAINING AND TESTING ###

lr = 0.001
num_epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

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


class skin_tone_accuracy:
    '''Computes accuracy for each skin tone class separately'''

    def __call__(self, outputs, labels, fst=None, ids=None):
        # outputs: [N, 6] logits, labels: [N] integer class indices (0-based)
        preds = outputs.argmax(dim=1)  # predicted classes [N]
        preds = preds.squeeze()
        labels = labels.squeeze()

        results = {}
        num_classes = outputs.size(1)

        for cls in range(num_classes):
            # Find indices where label == cls
            cls_mask = (labels == cls)
            total_cls = cls_mask.sum().item()

            if total_cls == 0:
                acc = float('nan')  # or 0.0 or skip
            else:
                correct_cls = (preds[cls_mask] == cls).sum().item()
                acc = 100.0 * correct_cls / total_cls

            results[f'class_{cls+1}_acc'] = acc  # classes labeled 1 to 6

        return 'skin_tone_acc', results



def summarise_metrics(metrics):
    output_lines = []

    # Overall accuracies
    accuracies = metrics['skin_tone_acc']

    for key, value in accuracies.items():
        output_lines.append(f'{key}: {value}')
      
    return output_lines

metrics = test_model(
    model,
    test_dataloader,
    device,
    skin_tone_accuracy()
)

experiment_data['Metrics'] = '\n'.join(summarise_metrics(metrics))


### MODEL EXPLANATION ###

model_gradCAM = UniversalGrad(model, 'layer4.2.conv3')
model_gradCAM.eval()
heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
grad_cam_path = save_plot_and_return_path(fig, 'gradCAM')
experiment_data['GradCAM Plot Path'] = grad_cam_path


### SAVE RESULTS ###

experiment_data['Train Dataset Visualisation'] = fig_train_path 
experiment_data['Test Dataset Visualisation'] = fig_test_path 
save_experiment_log(experiment_data, file_path=log_file)    


### INDIA DATA ###

df = pd.read_csv('Data/india_data/india_metadata_clean.csv')
zip_path = 'Data/india_data/india_images.zip'
zip_file = zipfile.ZipFile(zip_path, 'r')

name_map = {os.path.basename(name): name for name in zip_file.namelist()}

fitz_values = []

for i, img_name in enumerate(df['Image Name']):
    full_path_in_zip = name_map.get(img_name)
    if full_path_in_zip:
        try:
            with zip_file.open(full_path_in_zip) as file:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                input_tensor = transformations_val_test(image).unsqueeze(0).to(device)  
                
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = output.argmax(dim=1).item() + 1  

                fitz_values.append(predicted_class)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            fitz_values.append(None)
    else:
        print(f"Image {img_name} not found in ZIP")
        fitz_values.append(None)

df['Fitzpatrick'] = fitz_values

df.to_csv('india_metadata_estimate.csv', index=False)