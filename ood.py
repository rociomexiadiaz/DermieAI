import clip
import torch
from zip_dataset import *
from torchvision.transforms import transforms
from collections import defaultdict

# Load the model
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)


def clip_predict(idx, dataset:MultipleDatasets, text_prompts:list, random_crops=False, num_crops=5):

    sample = dataset[idx]  
    image = sample['image']
    size = image.size
    fst = sample['fst'].item() 

    if random_crops:
        text_inputs = torch.cat([clip.tokenize(c) for c in text_prompts]).to(device)
        cumulative_probs = defaultdict(float)
        transformations = transforms.RandomCrop(100)

        for i in range(num_crops):
            if min(size) > 100:
                image = transformations(image)
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
          
            for i, prompt in enumerate(text_prompts):
                cumulative_probs[prompt] += similarity[0, i].item()

        preds = {k: 100 * v / num_crops for k, v in cumulative_probs.items()}

    else:
    
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in text_prompts]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        k = min(5, text_features.shape[0])
        values, indices = similarity[0].topk(k)

        preds = {text_prompts[index]: 100*value.item() for value, index in zip(values, indices)}

    return preds, fst


def ood_performance(preds: list, fst: list):

    assert len(preds) == len(fst), "Length of preds and fst must match"

    predicted_labels = [max(p, key=p.get) for p in preds]
    unique_fst = sorted(set(fst))
    all_counts = {"skin": 0, "not skin": 0, "total": 0}

    for tone in unique_fst:
        tone_indices = [i for i, f in enumerate(fst) if f == tone]
        tone_preds = [predicted_labels[i] for i in tone_indices]

        skin_count = sum(1 for p in tone_preds if "human skin" in p)
        not_skin_count = sum(1 for p in tone_preds if "not skin" in p)
        total = len(tone_preds)

        all_counts["skin"] += skin_count
        all_counts["not skin"] += not_skin_count
        all_counts["total"] += total

        skin_pct = 100 * skin_count / total if total else 0
        not_skin_pct = 100 * not_skin_count / total if total else 0

        print(f"FST {tone}: Skin = {skin_pct:.2f}%, Not Skin = {not_skin_pct:.2f}% (N={total})")

    # Overall
    print("\n=== Overall ===")
    overall_skin_pct = 100 * all_counts["skin"] / all_counts["total"]
    overall_not_skin_pct = 100 * all_counts["not skin"] / all_counts["total"]
    print(f"Skin = {overall_skin_pct:.2f}%, Not Skin = {overall_not_skin_pct:.2f}% (N={all_counts['total']})")





def ood_performance2(preds: list, fst: list):

    assert len(preds) == len(fst), "Length of preds and fst must match"

    predicted_labels = [max(p, key=p.get) for p in preds]
    unique_fst = sorted(set(fst))
    all_counts = {"healthy skin": 0, "unhealthy skin": 0, "total": 0}

    for tone in unique_fst:
        tone_indices = [i for i, f in enumerate(fst) if f == tone]
        tone_preds = [predicted_labels[i] for i in tone_indices]

        healthy_skin_count = sum(1 for p in tone_preds if "human skin" in p)
        unhealthy_skin_count = sum(1 for p in tone_preds if "unhealthy skin" in p)
        total = len(tone_preds)

        all_counts["healthy skin"] += healthy_skin_count
        all_counts["unhealthy skin"] += unhealthy_skin_count
        all_counts["total"] += total

        skin_pct = 100 * healthy_skin_count / total if total else 0
        not_skin_pct = 100 * unhealthy_skin_count / total if total else 0

        print(f"FST {tone}: Healthy Skin = {skin_pct:.2f}%, Unhealthy Skin = {not_skin_pct:.2f}% (N={total})")

    # Overall
    print("\n=== Overall ===")
    overall_healthy_skin_pct = 100 * all_counts["healthy skin"] / all_counts["total"]
    overall_unhealthy_skin_pct = 100 * all_counts["unhealthy skin"] / all_counts["total"]
    print(f"Healthy Skin = {overall_healthy_skin_pct:.2f}%, Unhealthy Skin = {overall_unhealthy_skin_pct:.2f}% (N={all_counts['total']})")