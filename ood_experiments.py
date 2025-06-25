from zip_dataset import *
import torch
from metricsFunctions import *
from Baseline import *
from TestFunction import *
from xai import *
from ood import *

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17], transform=None) 
val_set = MultipleDatasets([dermie_metadata_val, pad_metadata_val, scin_metadata_val, fitz17_metadata_val], [images_dermie, images_pad, images_scin, images_fitz17], transform=None, diagnostic_encoder=train_set.diagnose_encoder)
test_set = MultipleDatasets([dermie_metadata_test, pad_metadata_test, scin_metadata_test, fitz17_metadata_test], [images_dermie, images_pad, images_scin, images_fitz17], transform=None, diagnostic_encoder=train_set.diagnose_encoder)


### OOD 1 ###
predictions = []
fsts = []
skin_indices = []

for i in range(len(train_set)):
    preds, fst = clip_predict(i, train_set, text_prompts=["a close-up of human skin", "not skin (background, objects, paper, clothes, etc.)"], random_crops=True)
    predictions.append(preds)
    predicted_label = max(preds, key=preds.get)
    if "human skin" in predicted_label:
        skin_indices.append(i)
    fsts.append(fst)

ood_performance(predictions, fsts)

### OOD 2 ###
predictions = []
fsts = []

for i in skin_indices:
    preds, fst = clip_predict(i, train_set, text_prompts=["a close-up of healthy, clean human skin", "a close-up of diseased, unhealthy skin (eczema, acne, rashes, etc.)"])
    predictions.append(preds)
    fsts.append(fst)

ood_performance2(predictions, fsts)



