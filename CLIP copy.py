import torch
from PIL import Image
import open_clip
from zip_dataset import *
from torchvision.transforms import transforms
from collections import defaultdict

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

lesion_model, _, lesion_preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
tokenizer = open_clip.get_tokenizer("ViT-L-14")

def clip_predict(idx, dataset:MultipleDatasets, text_prompts:list, random_crops=False, num_crops=5, model_name='LesionCLIP'):

    global lesion_model, lesion_preprocess, tokenizer

    model, preprocess = lesion_model, lesion_preprocess
    model.to(device)
    model.eval()

    sample = dataset[idx]  
    image = sample['image']
    condition = sample['condition']
    fst = sample['fst'].item() 

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


def ood_performance(preds: list, fst: list, conditions:list) -> list[str]:

    assert len(preds) == len(fst), "Length of preds and fst must match"

    predicted_labels = [max(p, key=p.get) for p in preds]
    unique_fst = sorted(set(fst))
    unique_labels = sorted(set(predicted_labels))

    # Initialize overall count dictionary
    all_counts = {label: 0 for label in unique_labels}
    all_counts["total"] = 0

    lines = []

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

    # Overall
    lines.append("\n=== Overall ===")
    for label in unique_labels:
        pct = 100 * all_counts[label] / all_counts["total"] if all_counts["total"] else 0
        lines.append(f"{label}: {pct:.2f}% (N={all_counts[label]})")

    return lines


