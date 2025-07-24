from zip_dataset import *
import torchvision.transforms as transforms
import torch
from VAE import *
from TestFunction import *
from metricsFunctions import *
import matplotlib.pyplot as plt
from xai import *
import datetime
import open_clip

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

lesion_clip_model, _, lesion_clip_preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")

class CLIPMultipleDatasets(Dataset):
    def __init__(self, metadata_list, image_zip_list, transform=None, diagnostic_encoder=None):
        """
        Args:
            metadata_list (list of pd.DataFrame): List of metadata DataFrames.
            image_zip_list (list of str): List of paths to image ZIP files (same order as metadata).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert len(metadata_list) == len(image_zip_list), "Metadata and image zip lists must be the same length"
        
        valid_pairs = [(metadata, zip_path) for metadata, zip_path in zip(metadata_list, image_zip_list) 
                            if metadata is not None and zip_path is not None]
        
        if not valid_pairs:
            raise ValueError("All datasets are None! No valid datasets to load.")
        
        # Separate back into lists
        valid_metadata_list = [pair[0] for pair in valid_pairs]
        valid_image_zip_list = [pair[1] for pair in valid_pairs]

        self.transform = transform
        self.zip_paths = valid_image_zip_list
        self.zips = [zipfile.ZipFile(p, 'r') for p in valid_image_zip_list]  
        self.metadata = pd.DataFrame()

        for i, df in enumerate(valid_metadata_list):
            df = df.copy()
            df['zip_index'] = i  
            self.metadata = pd.concat([self.metadata, df], ignore_index=True)

        self.img_ids = self.metadata['Image Name'].values
        self.zip_indices = self.metadata['zip_index'].values
        self.stratify_col = self.metadata['stratify_col'].values

        # Fitzpatrick encoding
        fst_encoder = OrdinalEncoder(categories=[['I', 'II', 'III', 'IV', 'V', 'VI']])
        self.fst_labels = fst_encoder.fit_transform(self.metadata['Fitzpatrick'].values.reshape(-1, 1)) + 1

        self.condition = self.metadata['Diagnosis'].values

        if diagnostic_encoder is None:
            self.diagnose_encoder = OneHotEncoder()
            self.diagnostic = self.diagnose_encoder.fit_transform(
                self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
        else:
            self.diagnose_encoder = diagnostic_encoder
            self.diagnostic = self.diagnose_encoder.transform(
                self.metadata['Diagnosis'].values.reshape(-1, 1)).toarray()
            
        # CLIP Embeddings
        global clip_model, clip_preprocess, lesion_clip_model, lesion_clip_preprocess, device
        self.clip_model, self.clip_preprocess = lesion_clip_model, lesion_clip_preprocess
        self.clip_model.to(device)
        self.clip_model.eval()


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_ids[idx]
        zip_idx = self.zip_indices[idx]
        zip_file = self.zips[zip_idx]

        try:
            with zip_file.open(img_name) as file:
                image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name} from zip {self.zip_paths[zip_idx]}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        image = self.clip_preprocess(image).unsqueeze(0).to(device)
    
        with torch.no_grad():
            image_embed = self.clip_model.encode_image(image)
        
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
    
        fst = torch.tensor(self.fst_labels[idx], dtype=torch.float)
        diagnosis = torch.tensor(self.diagnostic[idx], dtype=torch.float)

        return {
            'image': image_embed,
            'img_id': img_name,
            'fst': fst,
            'diagnosis': diagnosis,
            'condition': self.condition[idx]
        }


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

experiment_data['Datasets'] = 'Dermie + Padufes + SCIN + Fitzpatrick17k + India'


### CREATE DATASETS AND DATALOADERS ###

transformations = transforms.Compose([
    transforms.ToTensor(),  
    transforms.RandomAffine(degrees=10, shear= (-10,10,-10,10)),
    transforms.ToPILImage()
])

train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=transformations, clip=True, apply_augment=True) 
val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=None, diagnostic_encoder=train_set.diagnose_encoder)
test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test, india_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=None, diagnostic_encoder=train_set.diagnose_encoder)

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

    def forward(self, x):
        return self.fc(x)
    
encoder = FC()

model = VAEmodel(encoder=encoder, num_classes=num_conditions)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
resampler = AdaptiveResampler(alpha=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)


### MODEL TRAINING AND TESTING ###

model, fig = train_VAE(model, train_dataloader, val_dataloader, optimizer, scheduler, resampler, device=device, num_epochs=30)
loss_path = save_plot_and_return_path(fig, 'losses')

model = nn.Sequential(
    model.encoder,
    model.classifier,
   )

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

model_gradCAM = UniversalGrad(model, '0.layer4.2.conv3')
model_gradCAM.eval()
heatmaps, images_for_grad_cam, predicted_labels, real_labels = gradCAM(model_gradCAM, test_dataloader, device)
fig = visualize_gradcams_with_colorbars(images_for_grad_cam, heatmaps, predicted_labels, real_labels, conditions_mapping)
grad_cam_path = save_plot_and_return_path(fig, 'gradCAM')


### SAVE RESULTS ###

experiment_data['Train Dataset Visualisation'] = fig_train_path 
experiment_data['Test Dataset Visualisation'] = fig_test_path 
experiment_data['GradCAM Plot Path'] = grad_cam_path
save_experiment_log(experiment_data, file_path=log_file)
