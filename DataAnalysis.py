import re
import pandas as pd

def parse_combined_log(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # Split models by 'Python Filename:'
    model_blocks = re.split(r'(?=Python Filename:)', content)

    # Containers
    avg_sensitivities = []
    condition_sensitivities = []
    skin_tone_sensitivities = []
    stratified_sensitivities = []
    misclassifications = []
    misclassifications_by_tone = []

    for block in model_blocks:
        if not block.strip():
            continue

        # Extract model and dataset
        model = re.search(r'Python Filename:\s*(\S+)', block).group(1).replace('.py', '')
        datasets = re.search(r'Datasets:\s*(.+)', block).group(1)

        ### 1. Average Top-k Sensitivity
        for k in ['Top-1', 'Top-3', 'Top-5']:
            match = re.search(rf'Average {k} Sensitivity:\s*([\d.]+)%', block)
            if match:
                avg_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(match.group(1))
                })

        ### 2. Top-k Sensitivity per condition
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                condition_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })

        ### 3. Top-k Sensitivity per skin tone
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for tone_match in re.findall(rf'Skin Tone:\s*([\d.]+), {k} Accuracy:\s*([\d.]+)%', block):
                tone, value = tone_match
                skin_tone_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Skin Tone': float(tone),
                    'Metric': f'{k} Accuracy',
                    'Value': float(value)
                })

        ### 4. Stratified Top-k Sensitivity per skin tone and condition
        stratified_blocks = re.findall(r'Skin Tone:\s*([\d.]+)\n((?:\s+Condition:.*\n?)+)', block)
        for tone, section in stratified_blocks:
            matches = re.findall(r'Condition:\s*(.+?), (Top-[135]) Sensitivity:\s*([\d.]+)%', section)
            for condition, k, value in matches:
                stratified_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Skin Tone': float(tone),
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })

        ### 5. Number of misclassified samples
        misclassified = re.search(r'Number of misclassified samples:\s*(\d+)', block)
        if misclassified:
            misclassifications.append({
                'Model': model,
                'Datasets': datasets,
                'Misclassified Samples': int(misclassified.group(1))
            })

        ### 6. Most common misclassifications (overall)
        overall_misclass = re.findall(r'^\s*(\w+)\s*→\s*(\w+):\s*(\d+) times', block, re.MULTILINE)
        for true_class, pred_class, count in overall_misclass:
            misclassifications_by_tone.append({
                'Model': model,
                'Datasets': datasets,
                'Skin Tone': 'All',
                'True Class': true_class,
                'Predicted Class': pred_class,
                'Count': int(count)
            })

        ### 7. Most common misclassifications by skin tone
        tone_misclass_blocks = re.findall(r'Skin Tone ([\d.]+):\n((?:\s{2,}.+→.+:\s*\d+ times\n?)+)', block)
        for tone, section in tone_misclass_blocks:
            for true_class, pred_class, count in re.findall(r'\s*(\w+)\s*→\s*(\w+):\s*(\d+)', section):
                misclassifications_by_tone.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Skin Tone': float(tone),
                    'True Class': true_class,
                    'Predicted Class': pred_class,
                    'Count': int(count)
                })

    # Convert all to DataFrames
    return {
        'AverageSensitivities': pd.DataFrame(avg_sensitivities),
        'ConditionSensitivities': pd.DataFrame(condition_sensitivities),
        'SkinToneAccuracies': pd.DataFrame(skin_tone_sensitivities),
        'StratifiedSensitivities': pd.DataFrame(stratified_sensitivities),
        'MisclassifiedCounts': pd.DataFrame(misclassifications),
        'MisclassificationDetails': pd.DataFrame(misclassifications_by_tone)
    }


# Example usage:
log_path = "Results/combined_all.txt"
df_dict = parse_combined_log(log_path)

# Access example:
print(df_dict['AverageSensitivities'].head())
print(df_dict['StratifiedSensitivities'].head())
print(df_dict['SkinToneAccuracies'].head())
print(df_dict['StratifiedSensitivities'].head())
print(df_dict['MisclassifiedCounts'].head())
print(df_dict['MisclassificationDetails'].head())
