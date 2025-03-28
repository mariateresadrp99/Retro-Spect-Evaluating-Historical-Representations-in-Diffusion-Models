# utils.py

"""
Helper functions for predicting stylistic bias in generated images.
Includes style label mapping and a colorfulness metric for distinguishing black & white from color.
"""

import cv2
import numpy as np

# List of style class names used by the classifier
CLASS_NAMES = ["drawings", "engraving", "illustrations", "painting", "photography"]

def predict_colorfulness(image_path):
    """
    Compute the colorfulness score of an image based on Hasler & Süsstrunk (2003).
    Used to refine prediction when distinguishing between b&w and color photography.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)

    return std_root + (0.3 * mean_root)
