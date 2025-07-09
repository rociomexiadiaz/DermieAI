from zip_dataset import *
import torch
from metricsFunctions import *
from Baseline import *
from TestFunction import *
from xai import *
from CLIP import *

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

india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/india_data', 
                                                                                       images_dir='india_images.zip',
                                                                                       metadata_dir='india_metadata_final.csv',
                                                                                       stratification_strategy=stratification_strategy)

### DATASET COMBINATIONS ###

# All
all_train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17, images_india], transform=None) 

# Minus dermie
minusdermie_train_set = MultipleDatasets([pad_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_pad, images_scin, images_fitz17, images_india], transform=None) 

# Minus PADUFES
minuspad_train_set = MultipleDatasets([dermie_metadata_train, scin_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_scin, images_fitz17, images_india], transform=None) 

# Minus SCIN
minusscin_train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, fitz17_metadata_train, india_metadata_train], [images_dermie, images_pad, images_fitz17, images_india], transform=None) 

# Minus Fitz
minusfitz_train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, india_metadata_train], [images_dermie, images_pad, images_scin, images_india], transform=None) 

# Minus India
minusindia_train_set = MultipleDatasets([dermie_metadata_train, pad_metadata_train, scin_metadata_train, fitz17_metadata_train], [images_dermie, images_pad, images_scin, images_fitz17], transform=None) 

datasets = {'All': all_train_set,
            'Minus dermie': minusdermie_train_set,
            'Minus PADUFES': minuspad_train_set,
            'Minus SCIN': minusscin_train_set,
            'Minus Fitz': minusfitz_train_set,
            'Minus India': minusindia_train_set}


### EXPERIMENTS ###  

with open("ood_report.txt", "w") as f:    

    for dataset_name, dataset in datasets.items():

        f.write(dataset_name + "\n")

        predictions = []
        fsts = []
        skin_indices = []

        for i in range(len(dataset)):
            preds, fst = clip_predict(i, dataset, 
                                      text_prompts=["a close-up of human skin", "not skin (background, objects, paper, clothes, etc.)"], 
                                      random_crops=True,
                                      model='LesionCLIP')
            predictions.append(preds)
            predicted_label = max(preds, key=preds.get)
            if "human skin" in predicted_label:
                skin_indices.append(i)
            fsts.append(fst)

        lines = ood_performance(predictions, fsts)
        for line in lines:
            f.write(line + "\n")


        predictions = []
        fsts = []

        for i in skin_indices:
            preds, fst = clip_predict(i, dataset, 
                                      text_prompts=["a close-up of healthy, clean human skin", "a close-up of diseased, unhealthy skin (eczema, acne, rashes, etc.)"],
                                      random_crops=False,
                                      model='LesionCLIP')
            
            predictions.append(preds)
            fsts.append(fst)

        lines = ood_performance(predictions, fsts)
        for line in lines:
            f.write(line + "\n")



