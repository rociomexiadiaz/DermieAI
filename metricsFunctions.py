from collections import defaultdict, Counter

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


        #overall_common = Counter(
            #(v[1], v[2]) for v in misclassification.values()
        #).most_common(5)

        #skintone_common = {
            #tone: Counter(pairs).most_common(3)
            #for tone, pairs in misclassification_by_skin_tone.items()
        #}

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


def display_misclassifications(results, conditions_mapping): 
    output = []

    misclassified_samples = results.get('misclassified_samples', [])
    common_misclassifications = results.get('common_misclassifications', [])
    common_by_skintone = results.get('common_misclassifications_by_skintone', {})

    # Overall misclassified count
    output.append(f"\nNumber of misclassified samples: {len(misclassified_samples)}")

    # Most common misclassifications (overall)
    output.append("\n=== MOST COMMON MISCLASSIFICATIONS (OVERALL) ===")
    if hasattr(common_misclassifications, 'most_common'):
        # It's a Counter object
        for (true_label, pred_label), count in common_misclassifications.most_common():
            true_name = conditions_mapping[true_label]
            pred_name = conditions_mapping[pred_label]
            output.append(f"  {true_name} → {pred_name}: {count} times")
    else:
        # It's already a list of tuples
        for (true_label, pred_label), count in common_misclassifications:
            true_name = conditions_mapping[true_label]
            pred_name = conditions_mapping[pred_label]
            output.append(f"  {true_name} → {pred_name}: {count} times")

    # Most common misclassifications by skin tone
    output.append("\n=== MOST COMMON MISCLASSIFICATIONS BY SKIN TONE ===")
    for skintone, misclass_list in common_by_skintone.items():
        output.append(f"\nSkin Tone {skintone}:")
        if hasattr(misclass_list, 'most_common'):
            # It's a Counter object
            for (true_label, pred_label), count in misclass_list.most_common():
                true_name = conditions_mapping[true_label]
                pred_name = conditions_mapping[pred_label]
                output.append(f"  {true_name} → {pred_name}: {count} times")
        else:
            # It's already a list of tuples
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
    
    # Sensitivities
    sens_output = display_multi_k_sensitivity(metrics['multi_k_sens'], conditions_mapping, k_values)
    output_lines.extend(sens_output)
    
    # Stratified accuracies
    strat_acc_output = display_stratified_multi_k_accuracy(metrics['strat_multi_k_acc'], k_values)
    output_lines.extend(strat_acc_output)
    
    # Stratified sensitivities
    strat_sens_output = display_stratified_multi_k_sensitivity(metrics['strat_multi_k_sens'], conditions_mapping, k_values)
    output_lines.extend(strat_sens_output)
    
    # Misclassifications
    misclass_output = display_misclassifications(metrics['enhanced_misclassified'], conditions_mapping)
    output_lines.extend(misclass_output)

    return output_lines