import pandas as pd
import zipfile
import io
import os
from fst import get_sample_ita_kin  


df = pd.read_csv('Data/india_data/india_metadata_clean.csv')
zip_path = 'Data/india_data/india_images.zip'
zip_file = zipfile.ZipFile(zip_path, 'r')

name_map = {os.path.basename(name): name for name in zip_file.namelist()}

fitz_values = []

for img_name in df['Image Name']:
    full_path_in_zip = name_map.get(img_name)
    if full_path_in_zip:
        try:
            with zip_file.open(full_path_in_zip) as file:
                image_bytes = file.read()
                ita_label = get_sample_ita_kin(io.BytesIO(image_bytes))
                fitz_values.append(ita_label)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            fitz_values.append(None)
    else:
        print(f"Image {img_name} not found in ZIP")
        fitz_values.append(None)

df['Fitzpatrick'] = fitz_values

df.to_csv('india_metadata_estimate.csv', index=False)