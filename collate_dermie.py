import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np

def collate_metadata(metadata_files = ['Master Lesion Sheet - Sheet1.csv', 'More data - Sheet1.csv']):
    all_metadata = []

    for file in metadata_files:
        pd_metadata = pd.read_csv(f'dermie_images/{file}')

        if file == 'More data - Sheet1.csv':
            pd_metadata.rename(columns={'Diagnosis (most confident)':'Diagnosis'}, inplace=True)

        all_metadata.append(pd_metadata[['Image Name', 'Diagnosis', 'Fitzpatrick']])

    combined_metadata = pd.concat(all_metadata, ignore_index=True)

    return combined_metadata


class Diagnose:

    def __init__(self):

        self.condition_dictionary = {
            'Acne': ['acne'],
            'Actinic Keratosis':  ["actinic keratosis", "ak", "actinic keratoses"],
            'BCC': ['bcc', 'basal cell carcinoma'],
            'Contact Dermatitis': ['dermatitis', 'contact dermatitis'],
            'Cutaneous Lupus': ["cutaneous lupus", "discoid lupus", "subacute cutaneous lupus"],
            'Subcutaneous Sarcoidosis': ["cutaneous sarcoidosis", "sarcoidosis"],
            'Cyst': ['cyst', 'cysts'],
            'Eczema': ['eczema'],
            "Folliculitis": ["folliculitis", "scalp folliculitis"],
            "Lichen Planus": ["lichen planus", "lichenoid"],
            "Melanocytic Nevus": ["melanocytic nevus", "melanocytic naevus", "nevus", "naevus",
                                        "dermal nevus", "dermal naevus", "compound nevus", "compound naevus",
                                        "junctional nevus", "junctional naevus", "intradermal naevus",
                                        "congenital nevus", "congenital naevus", "benign nevus", "benign naevus"],
            "Melanoma": ["melanoma", "malignant melanoma", "acral melanoma", "nodular melanoma",
                                "lentigo maligna", "acral lentiginous melanoma"],
            "Psoriasis": ["psoriasis", "guttate psoriasis", "inverse psoriasis", 
                            "palmoplantar psoriasis", "pustulosis"],
            "SCC": ["scc", "squamous cell carcinoma", "keratoacanthoma", "ka"],
            "Urticaria": ["urticaria", "hives", "urtciaria", "urtcaria", "weals"],
            "Seborrheic Keratosis": ["seborrheic keratosis", "seborrhoiec keratosis", "seborheoic keratosis",
                                        "seborhoiec keratosis", "seb keratosis", "seb k", "seborrheic keratoses",
                                        "irritated seborrheic keratosis", "traumatised seb keratosis", 
                                        "seborhheic keratosis", "lichenoid keratosis", "lichenoid keratoiss"],
            "Seborrheic Dermatitis": ["seborrheic dermatitis"],
            "Vascular Lesions": ["haemangioma", "hemangioma", "cherry angioma", 
                                "angiokeratoma"]
                                
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
                return None
            
            else:
                return self.find_best_match(condition.casefold())
        else:
            return None
        




combined_metadata = collate_metadata()

# Drop duplicates
combined_metadata.drop_duplicates(subset=['Image Name'], inplace=True)

# Drop unlabelled fst
combined_metadata = combined_metadata[combined_metadata['Fitzpatrick'] != 'TODO']

# Map numeric fst
pd.set_option('future.no_silent_downcasting', True)
combined_metadata['Fitzpatrick'] = combined_metadata['Fitzpatrick'].replace({'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6})

# Clean diagnosis labels
classifier = Diagnose()
combined_metadata['Clean Diagnosis'] = combined_metadata['Diagnosis'].apply(lambda x: classifier.classify_label(x))

# Drop unknown diagnosis
combined_metadata.dropna(subset=['Clean Diagnosis'], inplace=True)

# Condition Distributions
print(combined_metadata['Clean Diagnosis'].value_counts())
combined_metadata['Clean Diagnosis'].value_counts().plot(kind='bar')
plt.show()

# Skin tone Distributions
fst_color_map = {
1: '#F5D5A0',
2: '#E4B589',
3: '#D1A479',
4: '#C0874F',
5: '#A56635',
6: '#4C2C27'
}

n_conditions = len(combined_metadata['Clean Diagnosis'].unique())
n_cols = 4  
n_rows = int(np.ceil(n_conditions / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

if n_rows == 1:
    axes = [axes] if n_cols == 1 else axes
else:
    axes = axes.flatten()

for i, condition in enumerate(combined_metadata['Clean Diagnosis'].unique()):
    counts = combined_metadata[combined_metadata['Clean Diagnosis'] == condition]['Fitzpatrick'].value_counts().sort_index()
    colors = [fst_color_map[fst] for fst in counts.index]

    axes[i].pie(counts.values, labels=counts.index, colors=colors, 
                autopct='%1.1f%%', startangle=90)
    axes[i].set_title(condition)
    axes[i].axis('equal')

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()




