from difflib import SequenceMatcher

class Diagnose:

    def __init__(self):

        self.condition_dictionary = {
            'acne': ['acne'],
            'actinic keratosis':  ["actinic keratosis", "ak", "actinic keratoses"],
            'BCC': ['bcc', 'basal cell carcinoma'],
            'contact dermatitis': ['dermatitis', 'contact dermatitis'],
            'cutaneous lupus': ["cutaneous lupus", "discoid lupus", "subacute cutaneous lupus"],
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
        