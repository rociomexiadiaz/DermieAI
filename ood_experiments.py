from zip_dataset import *
import torch
from metricsFunctions import *
from Baseline import *
from TestFunction import *
from xai import *
import open_clip
from collections import defaultdict

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
    

# Global variables for the model
lesion_model, _, lesion_preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
tokenizer = open_clip.get_tokenizer("ViT-L-14")

def clip_predict(idx, dataset, text_prompts: list, random_crops=False, num_crops=5, model_name='LesionCLIP'):
    """
    Predict using CLIP model on a single image from the dataset.
    
    Args:
        idx: Index of the image in the dataset
        dataset: MultipleDatasets object
        text_prompts: List of text prompts to compare against
        random_crops: Whether to use random crops (not implemented)
        num_crops: Number of crops to use (not implemented)
        model_name: Model name (currently only supports 'LesionCLIP')
    
    Returns:
        preds: Dictionary mapping text prompts to similarity scores
        fst: FST value from the sample
        condition: Actual condition/diagnosis from the dataset
    """
    global lesion_model, lesion_preprocess, tokenizer

    model, preprocess = lesion_model, lesion_preprocess
    model.to(device)
    model.eval()

    sample = dataset[idx]  
    image = sample['image']
    fst = sample['fst'].item() 
    
    # Extract the actual condition from the dataset - prioritize 'condition' field as requested
    condition = sample.get('condition', sample.get('diagnosis', 'unknown'))
    
    # Handle different data types for condition
    if hasattr(condition, 'item'):
        try:
            # If it's a single-element tensor, extract the scalar
            if condition.numel() == 1:
                condition = condition.item()
            else:
                # If it's a multi-element tensor, take the first element
                condition = condition[0].item() if hasattr(condition[0], 'item') else condition[0]
        except:
            # If extraction fails, convert to string
            condition = str(condition)
    elif isinstance(condition, (list, tuple)):
        # If it's a list or tuple, take the first element
        condition = condition[0] if len(condition) > 0 else 'unknown'
    
    # Ensure condition is a string
    condition = str(condition)
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        k = min(len(text_prompts), text_features.shape[0])
        values, indices = similarity[0].topk(k)

        preds = {text_prompts[index]: 100*value.item() for value, index in zip(values, indices)}

    return preds, fst, condition


def stage_performance(preds: list, fst: list, conditions: list, stage_name: str, expected_correct: str = None) -> list[str]:
    """
    Calculate performance metrics for each classification stage.
    
    Args:
        preds: List of prediction dictionaries
        fst: List of FST values
        conditions: List of actual conditions/diagnoses
        stage_name: Name of the classification stage
        expected_correct: For dermatology/lesion stages, what the correct answer should be
        
    Returns:
        lines: List of formatted result strings
    """
    assert len(preds) == len(fst) == len(conditions), "Length of preds, fst, and conditions must match"

    predicted_labels = [max(p, key=p.get) for p in preds]
    unique_fst = sorted(set(fst))
    unique_labels = sorted(set(predicted_labels))
    unique_conditions = sorted(set(conditions))

    # Initialize overall count dictionary
    all_counts = {label: 0 for label in unique_labels}
    all_counts["total"] = 0
    
    # Accuracy tracking
    stage_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_accuracy = {'correct': 0, 'total': 0}

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"STAGE: {stage_name}")
    lines.append(f"{'='*60}")

    # Calculate accuracy based on stage
    if expected_correct:  # For dermatology/lesion detection stages
        lines.append(f"\n=== {stage_name.upper()} DETECTION ACCURACY ===")
        lines.append(f"Expected correct answer: '{expected_correct}'")
        
        for i, pred_label in enumerate(predicted_labels):
            fst_val = fst[i]
            is_correct = expected_correct.lower() in pred_label.lower()
            
            # Update accuracy counters
            stage_accuracy[fst_val]['total'] += 1
            overall_accuracy['total'] += 1
            if is_correct:
                stage_accuracy[fst_val]['correct'] += 1
                overall_accuracy['correct'] += 1
    
    else:  # For condition classification
        lines.append(f"\n=== {stage_name.upper()} CLASSIFICATION ACCURACY ===")
        
        for i, (pred_label, actual_condition) in enumerate(zip(predicted_labels, conditions)):
            fst_val = fst[i]
            is_correct = False
            
            # Determine if prediction matches actual condition
            actual_condition_lower = str(actual_condition).lower()
            pred_label_lower = pred_label.lower()
            
            if 'malignant' in pred_label_lower:
                # Map actual conditions to malignant
                malignant_conditions = ['melanoma', 'scc', 'bcc', 'squamous cell carcinoma', 'basal cell carcinoma', 'malignant']
                is_correct = any(mal_cond in actual_condition_lower for mal_cond in malignant_conditions)
            elif 'benign' in pred_label_lower:
                # Map actual conditions to benign
                benign_conditions = ['nevus', 'mole', 'benign', 'keratosis', 'seborrheic']
                is_correct = any(ben_cond in actual_condition_lower for ben_cond in benign_conditions)
            elif 'eczema' in pred_label_lower:
                is_correct = 'eczema' in actual_condition_lower or 'dermatitis' in actual_condition_lower
            elif 'psoriasis' in pred_label_lower:
                is_correct = 'psoriasis' in actual_condition_lower
            else:
                # Direct string matching for other conditions
                is_correct = pred_label_lower in actual_condition_lower or actual_condition_lower in pred_label_lower
            
            # Update accuracy counters
            stage_accuracy[fst_val]['total'] += 1
            overall_accuracy['total'] += 1
            if is_correct:
                stage_accuracy[fst_val]['correct'] += 1
                overall_accuracy['correct'] += 1

    # Report accuracy by FST (skin tone)
    lines.append("\n--- Accuracy by Skin Tone (FST) ---")
    for fst_val in unique_fst:
        if stage_accuracy[fst_val]['total'] > 0:
            acc = 100 * stage_accuracy[fst_val]['correct'] / stage_accuracy[fst_val]['total']
            lines.append(f"FST {fst_val}: {acc:.1f}% ({stage_accuracy[fst_val]['correct']}/{stage_accuracy[fst_val]['total']})")
    
    # Overall accuracy
    if overall_accuracy['total'] > 0:
        overall_acc = 100 * overall_accuracy['correct'] / overall_accuracy['total']
        lines.append(f"\nOverall Accuracy: {overall_acc:.1f}% ({overall_accuracy['correct']}/{overall_accuracy['total']})")

    # CLIP prediction distribution by FST
    lines.append(f"\n=== {stage_name.upper()} PREDICTION DISTRIBUTION BY SKIN TONE ===")
    
    for tone in unique_fst:
        tone_indices = [i for i, f in enumerate(fst) if f == tone]
        tone_preds = [predicted_labels[i] for i in tone_indices]

        tone_counts = {label: 0 for label in unique_labels}
        for label in tone_preds:
            tone_counts[label] += 1

        total = len(tone_preds)
        all_counts["total"] += total
        for label in unique_labels:
            all_counts[label] += tone_counts[label]

        lines.append(f"\nFST {tone} (N={total}):")
        for label in unique_labels:
            pct = 100 * tone_counts[label] / total if total else 0
            lines.append(f"  {label}: {pct:.1f}% (N={tone_counts[label]})")

    # Overall prediction distribution
    lines.append(f"\n=== OVERALL {stage_name.upper()} PREDICTION DISTRIBUTION ===")
    for label in unique_labels:
        pct = 100 * all_counts[label] / all_counts["total"] if all_counts["total"] else 0
        lines.append(f"{label}: {pct:.1f}% (N={all_counts[label]})")

    # Show actual condition distribution (only for condition classification stage)
    if not expected_correct:
        lines.append(f"\n=== ACTUAL CONDITIONS IN DATASET ===")
        condition_counts = defaultdict(int)
        for condition in conditions:
            condition_counts[str(condition)] += 1
        
        lines.append("Overall condition distribution:")
        for condition, count in sorted(condition_counts.items()):
            pct = 100 * count / len(conditions)
            lines.append(f"  {condition}: {pct:.1f}% (N={count})")
        
        # Condition distribution by FST
        lines.append("\nCondition distribution by skin tone:")
        for tone in unique_fst:
            tone_indices = [i for i, f in enumerate(fst) if f == tone]
            tone_conditions = [conditions[i] for i in tone_indices]
            tone_condition_counts = defaultdict(int)
            for condition in tone_conditions:
                tone_condition_counts[str(condition)] += 1
            
            lines.append(f"\nFST {tone} (N={len(tone_conditions)}):")
            for condition, count in sorted(tone_condition_counts.items()):
                pct = 100 * count / len(tone_conditions) if len(tone_conditions) > 0 else 0
                lines.append(f"  {condition}: {pct:.1f}% (N={count})")

    return lines

### EXPERIMENTS ###  

with open("comprehensive_lesionclip_report.txt", "w") as f:    

    for dataset_name, dataset in datasets.items():

        f.write("="*80 + "\n")
        f.write(f"DATASET: {dataset_name}\n")
        f.write("="*80 + "\n")
        print(f"Processing {dataset_name}")

        # STAGE 1: Dermatology Detection
        print("Stage 1: Dermatology vs Non-dermatology classification...")
        predictions_dermato = []
        fsts = []
        conditions = []
        skin_indices = []

        for i in range(len(dataset)):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(dataset)} images")
                
            preds, fst, condition = clip_predict(i, dataset, 
                                      text_prompts=["dermatological image of human skin", 
                                                    "non-dermatological image with no visible skin"], 
                                      random_crops=False,
                                      model_name='LesionCLIP')
            predictions_dermato.append(preds)
            conditions.append(condition)
            predicted_label = max(preds, key=preds.get)
            if "human skin" in predicted_label:
                skin_indices.append(i)
            fsts.append(fst)

        lines = stage_performance(predictions_dermato, fsts, conditions, 
                                "Dermatology Detection", "dermatological image of human skin")
        for line in lines:
            f.write(line + "\n")

        # STAGE 2: Lesion Detection (only on images classified as dermatological)
        print(f"Stage 2: Lesion detection on {len(skin_indices)} dermatological images...")
        predictions_lesion = []
        fsts_lesion = []
        conditions_lesion = []
        lesion_indices = []

        for i in skin_indices:
            preds, fst, condition = clip_predict(i, dataset, 
                                      text_prompts=["skin with no visible lesions or abnormalities", 
                                                    "skin with visible lesions or abnormalities"],
                                      random_crops=False,
                                      model_name='LesionCLIP')
            
            predictions_lesion.append(preds)
            conditions_lesion.append(condition)
            fsts_lesion.append(fst)
            predicted_label = max(preds, key=preds.get)
            if "visible lesions" in predicted_label:
                lesion_indices.append(i)

        lines = stage_performance(predictions_lesion, fsts_lesion, conditions_lesion, 
                                "Lesion Detection", "skin with visible lesions or abnormalities")
        for line in lines:
            f.write(line + "\n")

        # STAGE 3: Condition Classification (only on images classified as having lesions)
        print(f"Stage 3: Condition classification on {len(lesion_indices)} lesion images...")
        predictions_condition = [] 
        conditions_condition = [] 
        fsts_condition = []

        if cancer:
            condition_prompts = ["malignant skin lesion (melanoma, scc, bcc, etc)", 
                               "benign neoplastic lesion (melanocytic nevus, etc)"]
        else:
            condition_prompts = ["eczema", "psoriasis"]

        for i in lesion_indices:
            preds, fst, condition = clip_predict(i, dataset, 
                                    text_prompts=condition_prompts,
                                    random_crops=False,
                                    model_name='LesionCLIP')
            
            predictions_condition.append(preds)
            conditions_condition.append(condition)
            fsts_condition.append(fst)

        lines = stage_performance(predictions_condition, fsts_condition, conditions_condition, 
                                "Condition Classification")
        for line in lines:
            f.write(line + "\n")

        # Summary
        f.write(f"\n{'='*80}\n")
        f.write(f"SUMMARY FOR {dataset_name}:\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total images: {len(dataset)}\n")
        f.write(f"Classified as dermatological: {len(skin_indices)} ({100*len(skin_indices)/len(dataset):.1f}%)\n")
        f.write(f"Classified as having lesions: {len(lesion_indices)} ({100*len(lesion_indices)/len(dataset):.1f}%)\n")
        f.write(f"Processed for condition classification: {len(predictions_condition)}\n")
        f.write(f"{'='*80}\n\n")

print("Comprehensive analysis complete. Results saved to 'comprehensive_lesionclip_report.txt'")