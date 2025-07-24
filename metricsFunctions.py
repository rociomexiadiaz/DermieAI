from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

class multi_k_accuracy():
    '''Returns the top k accuracy of model for multiple k values'''

    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        results = {}
        
        for k in self.k_values:
            ordered_k_outputs = outputs.argsort(dim=1, descending=True)[:, :k]  
            correct = (true_diagnosis[:, None] == ordered_k_outputs).any(dim=1)
            results[f'top_{k}_acc'] = correct.sum().float() / len(outputs) * 100.0
        
        return 'multi_k_acc', results


class multi_k_sensitivity():
    '''Returns the top k sensitivity (accuracy per skin condition) of model for multiple k values'''

    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        results = {}
        
        for k in self.k_values:
            ordered_k_outputs = outputs.argsort(dim=1, descending=True)[:, :k]
            sensitivities = {}

            for condition in true_diagnosis.unique():
                condition_indices = (true_diagnosis == condition).nonzero(as_tuple=True)[0]
            
                if len(condition_indices) == 0:
                    continue
                
                condition_true = true_diagnosis[condition_indices]
                condition_outputs = ordered_k_outputs[condition_indices]
                
                correct = (condition_true[:, None] == condition_outputs).any(dim=1)
                sensitivities[condition.item()] = correct.sum().float() / len(condition_indices) * 100.0

            results[f'top_{k}_sens'] = sensitivities
        
        return 'multi_k_sens', results
    

class stratified_multi_k_accuracy():
    '''Returns the stratified (by skin tone) top k accuracy of model for multiple k values'''

    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        results = {}
        
        for k in self.k_values:
            ordered_k_outputs = outputs.argsort(dim=1, descending=True)[:, :k]
            stratified_accuracy = defaultdict(float)
        
            for skintone in fst.unique():
                skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
                if len(skintone_indices) == 0:
                    continue
                
                fst_true = true_diagnosis[skintone_indices]
                fst_outputs = ordered_k_outputs[skintone_indices]

                correct = (fst_true[:, None] == fst_outputs).any(dim=1)
                stratified_accuracy[skintone.item()] = correct.sum().float() / len(skintone_indices) * 100.0
                
            results[f'strat_top_{k}_acc'] = stratified_accuracy
            
        return 'strat_multi_k_acc', results
    

class stratified_multi_k_sensitivity():
    '''Returns the stratified (by skin tone) top k sensitivity (accuracy per condition) of model for multiple k values'''

    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        results = {}
        
        for k in self.k_values:
            ordered_k_outputs = outputs.argsort(dim=1, descending=True)[:, :k]
            stratified_sensitivity = defaultdict(lambda: defaultdict(float))
        
            for skintone in fst.unique():
                skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
                if len(skintone_indices) == 0:
                    continue
                    
                fst_true = true_diagnosis[skintone_indices]
                fst_outputs = ordered_k_outputs[skintone_indices]

                for condition in fst_true.unique():
                    condition_indices = (fst_true == condition).nonzero(as_tuple=True)[0]
                    if len(condition_indices) == 0:
                        continue
                        
                    condition_true = fst_true[condition_indices]
                    condition_outputs = fst_outputs[condition_indices]
                   
                    correct = (condition_true[:, None] == condition_outputs).any(dim=1)
                    stratified_sensitivity[skintone.item()][condition.item()] = correct.sum().float() / len(condition_indices) * 100.0
                    
            results[f'strat_top_{k}_sens'] = stratified_sensitivity
            
        return 'strat_multi_k_sens', results


class enhanced_misclassified_samples():
    '''Returns misclassified samples and most common misclassifications'''

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)[:,0]

        misclassification = {}
        misclassification_by_skin_tone = defaultdict(list)
        
        for i in range(len(true_diagnosis)):
            if predicted_diagnosis[i] != true_diagnosis[i]:
                skin_tone = fst[i].item()
                misclassification[ids[i]] = [skin_tone, int(true_diagnosis[i]), int(predicted_diagnosis[i])]
                misclassification_by_skin_tone[skin_tone].append((int(true_diagnosis[i]), int(predicted_diagnosis[i])))


        overall_common = Counter(
            (v[1], v[2]) for v in misclassification.values()
        )

        skintone_common = {
            tone: Counter(pairs)
            for tone, pairs in misclassification_by_skin_tone.items()
        }
        
        sample_ids = set(misclassification.keys())

        results = {
            'misclassified_samples': sample_ids,
            'common_misclassifications': overall_common,
            'common_misclassifications_by_skintone': skintone_common
        }
        
        return 'enhanced_misclassified', results


class balanced_accuracy():
    '''Returns balanced accuracy for each condition (overall)'''

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1).cpu().numpy()
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)[:, 0].cpu().numpy()
        
        results = {}
        
        overall_balanced_acc = balanced_accuracy_score(true_diagnosis, predicted_diagnosis) * 100.0
        results['overall'] = overall_balanced_acc
        
        return 'balanced_accuracy', results


class stratified_balanced_accuracy():
    '''Returns balanced accuracy for each condition stratified by skin tone'''

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)[:, 0]
        
        results = {}
        
        for skintone in fst.unique():
            skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
            if len(skintone_indices) == 0:
                continue
                
            skintone_true = true_diagnosis[skintone_indices].cpu().numpy()
            skintone_pred = predicted_diagnosis[skintone_indices].cpu().numpy()
            
            # Overall balanced accuracy for this skin tone
            if len(skintone_true) > 0:
                overall_balanced_acc = balanced_accuracy_score(skintone_true, skintone_pred) * 100.0
                                
                results[skintone.item()] = {
                    'overall': overall_balanced_acc
                }
        
        return 'stratified_balanced_accuracy', results


class f1_score_metric():
    '''Returns F1 score for each condition (overall)'''

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1).cpu().numpy()
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)[:, 0].cpu().numpy()
        
        results = {}
        
        # Overall macro F1 score
        overall_f1 = f1_score(true_diagnosis, predicted_diagnosis, average='macro') * 100.0
        results['overall_macro'] = overall_f1
        
        # Overall weighted F1 score
        overall_f1_weighted = f1_score(true_diagnosis, predicted_diagnosis, average='weighted') * 100.0
        results['overall_weighted'] = overall_f1_weighted
        
        # F1 score per condition
        condition_f1 = {}
        for condition in np.unique(true_diagnosis):
            # Create binary classification: condition vs all others
            binary_true = (true_diagnosis == condition).astype(int)
            binary_pred = (predicted_diagnosis == condition).astype(int)
            
            if len(np.unique(binary_true)) > 1:  # Check if both classes are present
                f1 = f1_score(binary_true, binary_pred) * 100.0
                condition_f1[int(condition)] = f1
                
        results['per_condition'] = condition_f1
        
        return 'f1_score', results


class stratified_f1_score():
    '''Returns F1 score for each condition stratified by skin tone'''

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)[:, 0]
        
        results = {}
        
        for skintone in fst.unique():
            skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
            if len(skintone_indices) == 0:
                continue
                
            skintone_true = true_diagnosis[skintone_indices].cpu().numpy()
            skintone_pred = predicted_diagnosis[skintone_indices].cpu().numpy()
            
            if len(skintone_true) > 0:
                # Overall F1 scores for this skin tone
                overall_f1_macro = f1_score(skintone_true, skintone_pred, average='macro') * 100.0
                overall_f1_weighted = f1_score(skintone_true, skintone_pred, average='weighted') * 100.0
                
                # Per condition F1 score for this skin tone
                condition_f1 = {}
                for condition in np.unique(skintone_true):
                    # Create binary classification: condition vs all others
                    binary_true = (skintone_true == condition).astype(int)
                    binary_pred = (skintone_pred == condition).astype(int)
                    
                    if len(np.unique(binary_true)) > 1:  # Check if both classes are present
                        f1 = f1_score(binary_true, binary_pred) * 100.0
                        condition_f1[int(condition)] = f1
                
                results[skintone.item()] = {
                    'overall_macro': overall_f1_macro,
                    'overall_weighted': overall_f1_weighted,
                    'per_condition': condition_f1
                }
        
        return 'stratified_f1_score', results


# Updated display functions
def display_multi_k_sensitivity(sensitivities_dict, conditions_mapping, k_values=[1, 3, 5]):
    output = []
    
    for k in k_values:
        key = f'top_{k}_sens'
        if key in sensitivities_dict:
            output.append(f"\n=== TOP-{k} SENSITIVITY ===")
            sensitivities = sensitivities_dict[key]
            
            for condition in sensitivities.keys():
                try:
                    output.append(f"Condition: {conditions_mapping[int(condition)]}, Top-{k} Sensitivity: {sensitivities[int(condition)]:.2f}%")
                except KeyError:
                    output.append(f'Error: Condition {condition} not found in mapping')

            if sensitivities:
                average_sensitivity = sum(sensitivities.values()) / len(sensitivities)
                output.append(f"Average Top-{k} Sensitivity: {average_sensitivity:.2f}%")
    
    return output


def display_stratified_multi_k_sensitivity(sensitivities_dict, conditions_mapping, k_values=[1, 3, 5]):
    output = []
    
    for k in k_values:
        key = f'strat_top_{k}_sens'
        if key in sensitivities_dict:
            output.append(f"\n=== STRATIFIED TOP-{k} SENSITIVITY ===")
            sensitivities = sensitivities_dict[key]
            
            for skintone, condition_sensitivities in sensitivities.items():
                output.append(f"Skin Tone: {skintone}")
                for condition, sensitivity in condition_sensitivities.items():
                    try:
                        output.append(f"  Condition: {conditions_mapping[int(condition)]}, Top-{k} Sensitivity: {sensitivity:.2f}%")
                    except KeyError:
                        output.append(f'  Error: Condition {condition} not found in mapping')
                output.append("")
    
    return output


def display_stratified_multi_k_accuracy(accuracy_dict, k_values=[1, 3, 5]):
    output = []
    
    for k in k_values:
        key = f'strat_top_{k}_acc'
        if key in accuracy_dict:
            output.append(f"\n=== STRATIFIED TOP-{k} ACCURACY ===")
            accuracy = accuracy_dict[key]
            
            for skintone, acc in accuracy.items():
                output.append(f"Skin Tone: {skintone}, Top-{k} Accuracy: {acc:.2f}%")
    
    return output


def display_balanced_accuracy(balanced_acc_dict, conditions_mapping):
    output = []
    output.append(f"\n=== BALANCED ACCURACY ===")
    output.append(f"Overall Balanced Accuracy: {balanced_acc_dict['overall']:.2f}%")
    
    return output


def display_stratified_balanced_accuracy(strat_balanced_acc_dict, conditions_mapping):
    output = []
    output.append(f"\n=== STRATIFIED BALANCED ACCURACY ===")
    
    for skintone, metrics in strat_balanced_acc_dict.items():
        output.append(f"\nSkin Tone: {skintone}")
        output.append(f"  Overall Balanced Accuracy: {metrics['overall']:.2f}%")
        
    return output


def display_f1_score(f1_dict, conditions_mapping):
    output = []
    output.append(f"\n=== F1 SCORE ===")
    output.append(f"Overall F1 Score (Macro): {f1_dict['overall_macro']:.2f}%")
    output.append(f"Overall F1 Score (Weighted): {f1_dict['overall_weighted']:.2f}%")
    
    output.append("\nPer Condition F1 Score:")
    for condition, f1 in f1_dict['per_condition'].items():
        try:
            output.append(f"  {conditions_mapping[condition]}: {f1:.2f}%")
        except KeyError:
            output.append(f"  Condition {condition}: {f1:.2f}%")
    
    return output


def display_stratified_f1_score(strat_f1_dict, conditions_mapping):
    output = []
    output.append(f"\n=== STRATIFIED F1 SCORE ===")
    
    for skintone, metrics in strat_f1_dict.items():
        output.append(f"\nSkin Tone: {skintone}")
        output.append(f"  Overall F1 Score (Macro): {metrics['overall_macro']:.2f}%")
        output.append(f"  Overall F1 Score (Weighted): {metrics['overall_weighted']:.2f}%")
        output.append(f"  Per Condition:")
        for condition, f1 in metrics['per_condition'].items():
            try:
                output.append(f"    {conditions_mapping[condition]}: {f1:.2f}%")
            except KeyError:
                output.append(f"    Condition {condition}: {f1:.2f}%")
    
    return output


def display_sensitivity(sensitivity_dict, conditions_mapping):
    output = []
    output.append(f"\n=== SENSITIVITY (RECALL) ===")
    output.append(f"Overall Sensitivity (Macro): {sensitivity_dict['overall_macro']:.2f}%")
    output.append(f"Overall Sensitivity (Weighted): {sensitivity_dict['overall_weighted']:.2f}%")
    
    output.append("\nPer Condition Sensitivity:")
    for condition, sens in sensitivity_dict['per_condition'].items():
        try:
            output.append(f"  {conditions_mapping[condition]}: {sens:.2f}%")
        except KeyError:
            output.append(f"  Condition {condition}: {sens:.2f}%")
    
    return output


def display_stratified_sensitivity(strat_sensitivity_dict, conditions_mapping):
    output = []
    output.append(f"\n=== STRATIFIED SENSITIVITY (RECALL) ===")
    
    for skintone, metrics in strat_sensitivity_dict.items():
        output.append(f"\nSkin Tone: {skintone}")
        output.append(f"  Overall Sensitivity (Macro): {metrics['overall_macro']:.2f}%")
        output.append(f"  Overall Sensitivity (Weighted): {metrics['overall_weighted']:.2f}%")
        output.append(f"  Per Condition:")
        for condition, sens in metrics['per_condition'].items():
            try:
                output.append(f"    {conditions_mapping[condition]}: {sens:.2f}%")
            except KeyError:
                output.append(f"    Condition {condition}: {sens:.2f}%")
    
    return output


def display_misclassifications(results, conditions_mapping): 
    output = []

    misclassified_samples = results.get('misclassified_samples', [])
    common_misclassifications = results.get('common_misclassifications', [])
    common_by_skintone = results.get('common_misclassifications_by_skintone', {})

    # Overall misclassified count
    output.append(f"\nNumber of misclassified samples: {len(misclassified_samples)}")

    # Most common misclassifications (overall)
    output.append("\n=== MOST COMMON MISCLASSIFICATIONS (OVERALL) ===")
    for (true_label, pred_label), count in common_misclassifications:
        true_name = conditions_mapping[true_label]
        pred_name = conditions_mapping[pred_label]
        output.append(f"  {true_name} → {pred_name}: {count} times")

    # Most common misclassifications by skin tone
    output.append("\n=== MOST COMMON MISCLASSIFICATIONS BY SKIN TONE ===")
    for skintone, misclass_list in common_by_skintone.items():
        output.append(f"\nSkin Tone {skintone}:")
        for (true_label, pred_label), count in misclass_list:
            true_name = conditions_mapping[true_label]
            pred_name = conditions_mapping[pred_label]
            output.append(f"  {true_name} → {pred_name}: {count} times")

    return output


def summarise_enhanced_metrics(metrics, conditions_mapping, k_values=[1, 3, 5]):
    output_lines = []

    # Overall accuracies
    output_lines.append("=== OVERALL ACCURACIES ===")
    accuracies = metrics['multi_k_acc']
    for k in k_values:
        key = f'top_{k}_acc'
        if key in accuracies:
            output_lines.append(f"Top-{k} Accuracy: {accuracies[key]:.2f}%")
    
    # Multi-k sensitivities
    if 'multi_k_sens' in metrics:
        sens_output = display_multi_k_sensitivity(metrics['multi_k_sens'], conditions_mapping, k_values)
        output_lines.extend(sens_output)
    
    # Stratified accuracies
    if 'strat_multi_k_acc' in metrics:
        strat_acc_output = display_stratified_multi_k_accuracy(metrics['strat_multi_k_acc'], k_values)
        output_lines.extend(strat_acc_output)
    
    # Stratified multi-k sensitivities
    if 'strat_multi_k_sens' in metrics:
        strat_sens_output = display_stratified_multi_k_sensitivity(metrics['strat_multi_k_sens'], conditions_mapping, k_values)
        output_lines.extend(strat_sens_output)
    
    # NEW METRICS
    # Balanced Accuracy
    if 'balanced_accuracy' in metrics:
        balanced_acc_output = display_balanced_accuracy(metrics['balanced_accuracy'], conditions_mapping)
        output_lines.extend(balanced_acc_output)
    
    # Stratified Balanced Accuracy
    if 'stratified_balanced_accuracy' in metrics:
        strat_balanced_acc_output = display_stratified_balanced_accuracy(metrics['stratified_balanced_accuracy'], conditions_mapping)
        output_lines.extend(strat_balanced_acc_output)
    
    # F1 Score
    if 'f1_score' in metrics:
        f1_output = display_f1_score(metrics['f1_score'], conditions_mapping)
        output_lines.extend(f1_output)
    
    # Stratified F1 Score
    if 'stratified_f1_score' in metrics:
        strat_f1_output = display_stratified_f1_score(metrics['stratified_f1_score'], conditions_mapping)
        output_lines.extend(strat_f1_output)
    
    # Sensitivity
    if 'sensitivity' in metrics:
        sensitivity_output = display_sensitivity(metrics['sensitivity'], conditions_mapping)
        output_lines.extend(sensitivity_output)
    
    # Stratified Sensitivity
    if 'stratified_sensitivity' in metrics:
        strat_sensitivity_output = display_stratified_sensitivity(metrics['stratified_sensitivity'], conditions_mapping)
        output_lines.extend(strat_sensitivity_output)
    
    # Misclassifications
    if 'enhanced_misclassified' in metrics:
        misclass_output = display_misclassifications(metrics['enhanced_misclassified'], conditions_mapping)
        output_lines.extend(misclass_output)

    return output_lines