 ## Code from https://github.com/pbevan1/Detecting-Melanoma-Fairly/blob/main/preprocessing.py
## Paper: Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification

import cv2
from skimage import io, color
import math
import os
import numpy as np
from collections import Counter

# Hair removal for ITA calculation
def hair_remove(image):
    # Convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image

def resize_image(image, max_dim=224):
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def gray_world_correction(img):
    avg = img.mean(axis=(0, 1))
    scale = avg.mean() / avg
    corrected = (img * scale).clip(0, 255).astype(np.uint8)
    return corrected


debug_save_count = 0

def get_sample_ita_kin(path, debug_dir=None):
    DEBUG_SAVE_LIMIT = 20
    global debug_save_count

    ita_bnd_kin = -1
    try:
        rgb = io.imread(path)
        rgb = resize_image(rgb)
        #rgb = hair_remove(rgb)
        rgb = gray_world_correction(rgb)

        lab = color.rgb2lab(rgb)
        h, w = lab.shape[:2]
        patch_size = min(h, w) // 10  # roughly 10% of image dim

        # Define 8 central-ish patches (avoid edges and corners)
        # Positions chosen around center 112,112 ± some offset

        margin_h = int(h * 0.3)
        margin_w = int(w * 0.3)

        ys = [int(y) for y in np.linspace(margin_h, h - margin_h - patch_size, 3)]
        xs = [int(x) for x in np.linspace(margin_w, w - margin_w - patch_size, 3)]

        coords = [(y, x) for y in ys for x in xs]

        ita_lst = []

        debug_image = rgb.copy()

        for y, x in coords:
            # Make sure patch doesn't go out of bounds
            y1 = max(0, y)
            x1 = max(0, x)
            y2 = min(h, y1 + patch_size)
            x2 = min(w, x1 + patch_size)

            patch_L = lab[y1:y2, x1:x2, 0].mean()
            patch_b = lab[y1:y2, x1:x2, 2].mean()
            ita = math.atan((patch_L - 50) / patch_b) * (180 / math.pi)
            ita_lst.append(ita)
            
            def ita_to_fitz(ita):
                if ita > 55:
                    return 1
                elif 41 < ita <= 55:
                    return 2
                elif 28 < ita <= 41:
                    return 3
                elif 19 < ita <= 28:
                    return 4
                elif 10 < ita <= 19:
                    return 5
                else:
                    return 6

            if debug_dir:
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        fitz_classes = [ita_to_fitz(ita) for ita in ita_lst]
        ita_bnd_kin = Counter(fitz_classes).most_common(1)[0][0]

        if debug_dir!=None and (ita_bnd_kin == 6 or ita_bnd_kin == 1) and debug_save_count < DEBUG_SAVE_LIMIT:
            os.makedirs('debugs', exist_ok=True)
        
            debug_path = os.path.join('debugs', f"{debug_dir}_patches.png")
            cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

            debug_save_count += 1

    except Exception:
        pass

    return ita_bnd_kin  