import clip
import torch
from zip_dataset import *
from torchvision.transforms import transforms
from collections import defaultdict
from collections import Counter


# Load the model
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)


FITZPATRICK_PROMPTS = [
    "very pale skin, blonds or gingers",
    "pale skin tone, almost never tans", 
    "medium skin tone, tans sometimes",
    "olive skin tone or indian",
    "brown skin tone or dark indian",
    "dark brown/black skin tone"
]


def clip_fitzpatrick_predict(idx, dataset: MultipleDatasets, text_prompts = FITZPATRICK_PROMPTS, random_crops=False, num_crops=5):
    """
    Predict Fitzpatrick skin tone using CLIP
    
    Args:
        idx: Dataset index
        dataset: MultipleDatasets object
        prompt_style: "detailed", "short", or "numeric" - determines which prompts to use
        random_crops: Whether to use random crops for ensemble prediction
        num_crops: Number of random crops to use if random_crops=True
    
    Returns:
        preds: Dictionary mapping Fitzpatrick types (1-6) to confidence scores
        fst: Ground truth Fitzpatrick skin tone
    """

    sample = dataset[idx]  
    image = sample['image']
    image_name = sample['img_id']
    size = image.size
    fst = sample['fst'].item() 

    if random_crops:
        text_inputs = torch.cat([clip.tokenize(c) for c in text_prompts]).to(device)
        cumulative_probs = defaultdict(float)
        transformations = transforms.RandomCrop(100)

        for i in range(num_crops):
            crop_image = image
            if min(size) > 100:
                crop_image = transformations(image)
            image_input = preprocess(crop_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
          
            for j, prompt in enumerate(text_prompts):
                cumulative_probs[j+1] += similarity[0, j].item()  # Store as Fitzpatrick type 1-6

        # Convert to Fitzpatrick type predictions (1-6)
        preds = {k: 100 * v / num_crops for k, v in cumulative_probs.items()}

    else:
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(c) for c in text_prompts]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Convert to Fitzpatrick type predictions (1-6)
        preds = {i+1: 100*similarity[0, i].item() for i in range(len(text_prompts))}

    return preds, fst, image_name

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/india_data', 
                                                                                       images_dir='india_images.zip',
                                                                                       metadata_dir='india_meta_valid_images.csv',
                                                                                       stratification_strategy='Diagnosis')

dataset = MultipleDatasets([india_metadata_train, india_metadata_val, india_metadata_test], [images_india, images_india, images_india], transform=None) 

india_metadata_with_preds = pd.read_csv('Data/india_data/india_meta_valid_images.csv')

for i in range(len(dataset)):
    if i % 500 == 0 or i == len(dataset) - 1:
        print(f"Processing sample {i+1}/{len(dataset)}")
    preds, fst, image_name = clip_fitzpatrick_predict(i, dataset)
    image_name = os.path.basename(image_name)
    predicted_type = max(preds, key=preds.get)
    india_metadata_with_preds.loc[india_metadata_with_preds['Image Name'] == image_name, 'Fitzpatrick'] = predicted_type

india_metadata_with_preds.to_csv('india_metadata_with_predicted_fst.csv', index=False)
   
