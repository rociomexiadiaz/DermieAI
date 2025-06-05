import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np
import os

def collate_metadata(metadata_files = ['Labelled_images_for_training_700_15_disease.csv']):
    all_metadata = []

    for file in metadata_files:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_dir, 'Data/dermie_images')
        pd_metadata = pd.read_csv(f'{path}/{file}')

        all_metadata.append(pd_metadata)

    combined_metadata = pd.concat(all_metadata, ignore_index=True)

    return combined_metadata


class Diagnose:

    def __init__(self):

        self.condition_dictionary = {
            'acne': ['acne'],
            'actinic Keratosis':  ["actinic keratosis", "ak", "actinic keratoses"],
            'BCC': ['bcc', 'basal cell carcinoma'],
            'contact cermatitis': ['dermatitis', 'contact dermatitis'],
            'cutaneous cupus': ["cutaneous lupus", "discoid lupus", "subacute cutaneous lupus"],
            'cutaneous sarcoidosis': ["cutaneous sarcoidosis", "sarcoidosis"],
            'cyst': ['cyst', 'cysts'],
            'eczema': ['eczema'],
            "folliculitis": ["folliculitis", "scalp folliculitis"],
            "lichen planus": ["lichen planus", "lichenoid"],
            "melanocytic nevus": ["melanocytic nevus", "melanocytic naevus", "nevus", "naevus",
                                        "dermal nevus", "dermal naevus", "compound nevus", "compound naevus",
                                        "junctional nevus", "junctional naevus", "intradermal naevus",
                                        "congenital nevus", "congenital naevus", "benign nevus", "benign naevus"],
            "melanoma": ["melanoma", "malignant melanoma", "acral melanoma", "nodular melanoma",
                                "lentigo maligna", "acral lentiginous melanoma"],
            "psoriasis": ["psoriasis", "guttate psoriasis", "inverse psoriasis", 
                            "palmoplantar psoriasis", "pustulosis"],
            "SCC": ["scc", "squamous cell carcinoma", "keratoacanthoma", "ka"],
            "urticaria": ["urticaria", "hives", "urtciaria", "urtcaria", "weals"]
                                }
        
    def find_best_match(self, condition:str, threshold=0.7):

        best_score = 0
        best_standard = None
        
        for name, variations in self.condition_dictionary.items():
            for variation in variations:
                score = SequenceMatcher(None, condition, variation).ratio()
                
                if variation in condition:
                    score = max(score, 0.8)  
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_standard = name
        
        return best_standard
    
    def classify_label(self, condition: str):

        if isinstance(condition, str): 
            if '?' in condition:
                return 'NULL'
            
            else:
               
                return self.find_best_match(condition.casefold())
             
        else:
            return 'NULL'
        

        

        




combined_metadata = collate_metadata()

# Drop duplicates
combined_metadata.drop_duplicates(subset=['Image Name'], inplace=True)

# Drop unlabelled fst
combined_metadata = combined_metadata[combined_metadata['Fitzpatrick'] != 'TODO']

# Clean diagnosis labels
classifier = Diagnose()
combined_metadata['Clean Diagnosis'] = combined_metadata['Diagnosis'].apply(lambda x: classifier.classify_label(x))
combined_metadata['Clean Diagnosis'].replace({None:'NULL', np.nan:'NULL'},inplace=True)
combined_metadata['15_diseases'].replace({None:'NULL', np.nan:'NULL'}, inplace=True)
pd.set_option('display.max_rows', None)
print((combined_metadata[combined_metadata["Clean Diagnosis"] != combined_metadata['15_diseases']][['Diagnosis','Clean Diagnosis','15_diseases']]))
print(len(combined_metadata[combined_metadata["Clean Diagnosis"] != combined_metadata['15_diseases']][['Clean Diagnosis','15_diseases']]))