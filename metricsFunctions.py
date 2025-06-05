from collections import defaultdict

class top_k_accuracy():
    '''Returns the top k accuracy of model '''

    def __init__(self, k):
        self.k = k

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        ordered_k_outputs = outputs.argsort(dim=1, descending=True)[:, :self.k]  

        correct = (true_diagnosis[:, None] == ordered_k_outputs).any(dim=1)
        
        return f'top_{self.k}_acc', correct.sum().float() / len(outputs) * 100.0


class top_k_sensitivity():
    '''Returns the top k sensitivity of model '''

    def __init__(self, k):
        self.k = k

    def __call__(self, outputs, labels, fst, ids):
        top_k_acc = top_k_accuracy(self.k)

        true_diagnosis = labels.argmax(dim=1)
    
        sensitivities = {}

        for condition in true_diagnosis.unique():
            condition_indices = (true_diagnosis == condition).nonzero(as_tuple=True)[0]
        
            if len(condition_indices) == 0:
                continue
            
            condition_labels = labels[condition_indices]
            condition_outputs = outputs[condition_indices]
            
            correct = top_k_acc(condition_outputs, condition_labels, None, None)
            
            sensitivities[condition.item()] = correct[1] 

        return f'top_{self.k}_sens', sensitivities
    

class stratified_k_accuracy():
    '''Returns the stratified top k accuracy of model '''

    def __init__(self, k):
        self.k = k

    def __call__(self, outputs, labels, fst, ids):
        stratified_accuracy = defaultdict(float)
        top_k_acc = top_k_accuracy(self.k)
    
        for skintone in fst.unique():
            skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
            if len(skintone_indices) == 0:
                continue
            fst_labels = labels[skintone_indices]
            fst_outputs = outputs[skintone_indices]

            correct = top_k_acc(fst_outputs, fst_labels, None, None)
            stratified_accuracy[skintone.item()] = correct[1].item() 
        return f'strat_top_{self.k}_acc', stratified_accuracy
    

class stratified_k_sensitivity():
    '''Returns the stratified top k sensitivity of model '''

    def __init__(self, k):
        self.k = k

    def __call__(self, outputs, labels, fst, ids):
        top_k_acc = top_k_accuracy(self.k)
        stratified_sensitivity = defaultdict(lambda: defaultdict(float))
    
        for skintone in fst.unique():
            skintone_indices = (fst == skintone).nonzero(as_tuple=True)[0]
            if len(skintone_indices) == 0:
                continue
            fst_labels = labels[skintone_indices]
            true_diagnosis = fst_labels.argmax(dim=1)
            fst_outputs = outputs[skintone_indices]

            for condition in true_diagnosis.unique():
                condition_indices = (true_diagnosis == condition).nonzero(as_tuple=True)[0]
                if len(condition_indices) == 0:
                    continue
                condition_labels = fst_labels[condition_indices]
                condition_outputs = fst_outputs[condition_indices]
                correct = top_k_acc(condition_outputs, condition_labels, None, None)
                stratified_sensitivity[skintone][condition] = correct[1].item() 
        return f'strat_top_{self.k}_sens', stratified_sensitivity
    

class missclassified_samples():

    def __call__(self, outputs, labels, fst, ids):
        true_diagnosis = labels.argmax(dim=1)
        predicted_diagnosis = outputs.argsort(dim=1, descending=True)

        sample_ids = []

        for i in range(len(true_diagnosis)):
            if true_diagnosis[i] not in predicted_diagnosis[i, 0:3]:
                sample_ids.append(ids[i])
        
        return 'missclassified_samples', sample_ids

    
def display_top3_sensitivity(sensitivities, conditions_mapping):
    for condition in sensitivities.keys():
        try:
            print(f"Condition: {conditions_mapping[int(condition)]}, Top-3 Sensitivity: {sensitivities[int(condition)]:.2f}%")
        except KeyError:
            print('Error: Condition not found in mapping:')

    average_sensitivity = sum(sensitivities.values()) / len(sensitivities)
    print(f"Average Top-3 Sensitivity: {average_sensitivity:.2f}%")

def display_stratified_sensitivity(sensitivities, conditions_mapping):
    for skintone, condition_sensitivities in sensitivities.items():
        print(f"Skin Tone: {skintone}")
        for condition, sensitivity in condition_sensitivities.items():
            try:
                print(f"  Condition: {conditions_mapping[int(condition)]}, Top-3 Sensitivity: {sensitivity:.2f}%")
            except KeyError:
                print('  Error: Condition not found in mapping:')
        print()

def display_stratified_accuracy(accuracy):
    for skintone, acc in accuracy.items():
        print(f"Skin Tone: {skintone}, Top-3 Accuracy: {acc:.2f}%")

def summarise_metrics(metrics, conditions_mapping):

    if 'top_3_acc' in metrics:
        print(f"Top-3 Accuracy: {metrics['top_3_acc']:.2f}%")
    if 'top_3_sens' in metrics:
        display_top3_sensitivity(metrics['top_3_sens'], conditions_mapping)
    if 'strat_top_3_acc' in metrics:
        display_stratified_accuracy(metrics['strat_top_3_acc'])
    if 'strat_top_3_sens' in metrics:
        display_stratified_sensitivity(metrics['strat_top_3_sens'], conditions_mapping)
    if 'missclassified_samples' in metrics:
        print(f"Number of missclassified samples: {len(metrics['missclassified_samples'])}")


   