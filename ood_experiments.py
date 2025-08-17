from zip_dataset import *
import torch
from metricsFunctions import *
from Baseline import *
from TestFunction import *
from xai import *
from CLIP import *


cancer = True

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

seed = 42

### LOAD DATA ###

stratification_strategy = 'Diagnosis'  # 'stratify_col' -> Ensure all conditions and skin tones are in both train and test

dermie_metadata_train, dermie_metadata_test, dermie_metadata_val, images_dermie = load_dataset(project_dir=project_dir,
                                                                                               path_folder=r'Data/dermie_data', 
                                                                                               images_dir='master_data_june_7_2025.zip',
                                                                                               metadata_dir='master_data_june_7_2025.csv',
                                                                                               stratification_strategy=stratification_strategy,
                                                                                               seed=seed)

pad_metadata_train, pad_metadata_test, pad_metadata_val, images_pad = load_dataset(project_dir=project_dir,
                                                                                   path_folder=r'Data/padufes', 
                                                                                   images_dir='padufes_images.zip',
                                                                                   metadata_dir='padufes_metadata_new.csv',
                                                                                   stratification_strategy=stratification_strategy,
                                                                                   seed=seed)

scin_metadata_train, scin_metadata_test, scin_metadata_val, images_scin = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/scin', 
                                                                                       images_dir='scin_images.zip',
                                                                                       metadata_dir='scin_metadata_new.csv',
                                                                                       stratification_strategy=stratification_strategy,
                                                                                       seed=seed)

fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/fitz17k', 
                                                                                       images_dir='fitzpatrick17k_images.zip',
                                                                                       metadata_dir='fitzpatrick17k_metadata_new.csv',
                                                                                       stratification_strategy=stratification_strategy,
                                                                                       seed=seed)

### DATASET COMBINATIONS ###


dermie = MultipleDatasets([dermie_metadata_train, dermie_metadata_val, dermie_metadata_test], [images_dermie, images_dermie, images_dermie], transform=None) 

if cancer:
    pad = MultipleDatasets([pad_metadata_train, pad_metadata_val, pad_metadata_test], [images_pad, images_pad, images_pad], transform=None) 

scin = MultipleDatasets([scin_metadata_train, scin_metadata_val, scin_metadata_test], [images_scin, images_scin, images_scin], transform=None) 

fitz17 = MultipleDatasets([fitz17_metadata_train, fitz17_metadata_val, fitz17_metadata_test], [images_fitz17, images_fitz17, images_fitz17], transform=None) 

if cancer:
    datasets = {'Dermie': dermie,
                'PADUFES': pad,
                'SCIN': scin,
                'Fitzpatrick17k': fitz17,
                }
    
else: 
    datasets = {'Dermie': dermie,
                'SCIN': scin,
                'Fitzpatrick17k': fitz17,
                }

### EXPERIMENTS ###  

with open("ood_report.txt", "w") as f:    

    for dataset_name, dataset in datasets.items():

        f.write(dataset_name + "\n")
        print(dataset_name)

        predictions = []
        fsts = []
        conditions = []
        skin_indices = []

        for i in range(len(dataset)):
            preds, fst, condition = clip_predict(i, dataset, 
                                      text_prompts=["dermatological image of human skin", 
                                                    "non-dermatological image with no visible skin"], 
                                      random_crops=False,
                                      model_name='LesionCLIP')
            predictions.append(preds)
            conditions.append(condition)
            predicted_label = max(preds, key=preds.get)
            if "human skin" in predicted_label:
                skin_indices.append(i)
            fsts.append(fst)

        lines = ood_performance(predictions, fsts, conditions)
        for line in lines:
            f.write(line + "\n")


        predictions = []
        fsts = []
        conditions = []

        for i in skin_indices:
            preds, fst, condition = clip_predict(i, dataset, 
                                      text_prompts=["skin with no visible lesions or abnormalities", 
                                                    "skin with visible lesions or abnormalities"],
                                      random_crops=False,
                                      model_name='LesionCLIP')
            
            predictions.append(preds)
            conditions.append(condition)
            fsts.append(fst)

        lines = ood_performance(predictions, fsts, conditions)
        for line in lines:
            f.write(line + "\n")

        predictions = [] 
        conditions = [] 
        fsts = []

        if cancer:
            for i in skin_indices:
                preds, fst, condition = clip_predict(i, dataset, 
                                        text_prompts=["malignant skin lesion (melanoma, scc, bcc, etc)", 
                                                      "benign neoplastic lesion (melanocytic nevus, etc)"],
                                        random_crops=False,
                                        model_name='LesionCLIP')
                
                predictions.append(preds)
                conditions.append(condition)
                fsts.append(fst)

            lines = ood_performance(predictions, fsts, conditions)
            for line in lines:
                f.write(line + "\n")

        else:
            for i in skin_indices:
                preds, fst, condition = clip_predict(i, dataset, 
                                        text_prompts=["eczema", "psoriasis"],
                                        random_crops=False,
                                        model_name='LesionCLIP')
                
                predictions.append(preds)
                conditions.append(condition)
                fsts.append(fst)

            lines = ood_performance(predictions, fsts, conditions)
            for line in lines:
                f.write(line + "\n")



