import torch
from PIL import Image
import open_clip
from zip_dataset import *
from torchvision.transforms import transforms
from collections import defaultdict

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    # Extract the actual condition/diagnosis from the dataset
    condition = sample.get('diagnosis', sample.get('condition', 'unknown'))
    
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
        k = min(5, text_features.shape[0])
        values, indices = similarity[0].topk(k)

        preds = {text_prompts[index]: 100*value.item() for value, index in zip(values, indices)}

    return preds, fst, condition


def ood_performance(preds: list, fst: list, conditions: list) -> list[str]:
    """
    Calculate out-of-distribution performance metrics including condition accuracy.
    
    Args:
        preds: List of prediction dictionaries
        fst: List of FST values
        conditions: List of actual conditions/diagnoses
        
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
    
    # Condition accuracy tracking
    condition_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_accuracy = {'correct': 0, 'total': 0}

    lines = []

    # Calculate condition accuracy for diagnosis stages
    if any(keyword in unique_labels[0].lower() for keyword in ['malignant', 'benign', 'eczema', 'psoriasis']):
        lines.append("\n=== CONDITION ACCURACY ===")
        
        for i, (pred_label, actual_condition) in enumerate(zip(predicted_labels, conditions)):
            fst_val = fst[i]
            is_correct = False
            
            # Check if prediction matches actual condition
            if 'malignant' in pred_label.lower():
                # Map actual conditions to malignant/benign
                malignant_conditions = ['melanoma', 'scc', 'bcc', 'squamous cell carcinoma', 'basal cell carcinoma']
                is_correct = any(mal_cond in str(actual_condition).lower() for mal_cond in malignant_conditions)
            elif 'benign' in pred_label.lower():
                benign_conditions = ['nevus', 'mole', 'benign']
                is_correct = any(ben_cond in str(actual_condition).lower() for ben_cond in benign_conditions)
            elif 'eczema' in pred_label.lower():
                is_correct = 'eczema' in str(actual_condition).lower() or 'dermatitis' in str(actual_condition).lower()
            elif 'psoriasis' in pred_label.lower():
                is_correct = 'psoriasis' in str(actual_condition).lower()
            
            # Update accuracy counters
            condition_accuracy[fst_val]['total'] += 1
            overall_accuracy['total'] += 1
            if is_correct:
                condition_accuracy[fst_val]['correct'] += 1
                overall_accuracy['correct'] += 1

        # Report accuracy by FST
        for fst_val in unique_fst:
            if condition_accuracy[fst_val]['total'] > 0:
                acc = 100 * condition_accuracy[fst_val]['correct'] / condition_accuracy[fst_val]['total']
                lines.append(f"FST {fst_val} Diagnostic Accuracy: {acc:.1f}% ({condition_accuracy[fst_val]['correct']}/{condition_accuracy[fst_val]['total']})")
        
        # Overall accuracy
        if overall_accuracy['total'] > 0:
            overall_acc = 100 * overall_accuracy['correct'] / overall_accuracy['total']
            lines.append(f"Overall Diagnostic Accuracy: {overall_acc:.1f}% ({overall_accuracy['correct']}/{overall_accuracy['total']})")

    # Standard OOD performance by FST
    lines.append("\n=== CLIP PREDICTION DISTRIBUTION ===")
    
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
            lines.append(f"  {label}: {pct:.2f}%")

    # Overall distribution
    lines.append("\n=== Overall Distribution ===")
    for label in unique_labels:
        pct = 100 * all_counts[label] / all_counts["total"] if all_counts["total"] else 0
        lines.append(f"{label}: {pct:.2f}% (N={all_counts[label]})")

    # Show actual condition distribution
    lines.append(f"\n=== Actual Conditions in Dataset ===")
    condition_counts = defaultdict(int)
    for condition in conditions:
        condition_counts[str(condition)] += 1
    
    for condition, count in sorted(condition_counts.items()):
        pct = 100 * count / len(conditions)
        lines.append(f"{condition}: {pct:.1f}% (N={count})")

    return lines