## Code from https://github.com/pbevan1/Detecting-Melanoma-Fairly/blob/main/preprocessing.py
## Paper: Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification

import cv2
from skimage import io, color
import math

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


# Calculates Fitzpatrick skin type of an image using Kinyanjui et al.'s thresholds
def get_sample_ita_kin(path):
    ita_bnd_kin = -1
    try:
        rgb = io.imread(path)
        rgb = hair_remove(rgb)
        lab = color.rgb2lab(rgb)
        h, w = lab.shape[:2]
        patch_size = 20

        # Define 8 central-ish patches (avoid edges and corners)
        # Positions chosen around center 112,112 ± some offset

        coords = [
            (90, 90),    # upper-left center
            (90, 112),   # upper-middle center
            (90, 134),   # upper-right center

            (112, 90),   # middle-left center
            (112, 112),  # exact center
            (112, 134),  # middle-right center

            (134, 90),   # lower-left center
            (134, 112),  # lower-middle center
        ]

        ita_lst = []
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

        ita_max = max(ita_lst)

        if ita_max > 55:
            ita_bnd_kin = 1
        elif 41 < ita_max <= 55:
            ita_bnd_kin = 2
        elif 28 < ita_max <= 41:
            ita_bnd_kin = 3
        elif 19 < ita_max <= 28:
            ita_bnd_kin = 4
        elif 10 < ita_max <= 19:
            ita_bnd_kin = 5
        else:
            ita_bnd_kin = 6

    except Exception:
        pass

    return ita_bnd_kin