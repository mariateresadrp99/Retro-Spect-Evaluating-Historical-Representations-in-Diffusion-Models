#!/usr/bin/env python3
"""
predict_visual_style.py

A self-contained script to:
 1. Load metadata for the HistVis synthetic-image dataset.
 2. Download and load the fine-tuned CNN style‐predictor from Hugging Face.
 3. Predict a base visual style for each image.
 4. If the base style is “photography,” compute a colorfulness score 
    and classify into “photography (monochrome)” vs. “photography (color)”
    according to a user-specified threshold.
 5. Save a CSV with one row per image, including:
      - image_path
      - base_style
      - detailed_style
      - colorfulness (float or None)
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2  # OpenCV for colorfulness metric
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------------------------------------------------------------------------
# SECTION 1: CONSTANTS & HELPER FUNCTIONS
# ------------------------------------------------------------------------------
CLASS_NAMES = [
    "drawings",
    "engravings",
    "illustrations",
    "paintings",
    "photography"
]

def predict_colorfulness(img_path: str) -> float:
    """
    Compute the Hasler–Süsstrunk colorfulness metric for an image file.
    Args:
        img_path: Path to an image on disk.
    Returns:
        A non-negative float. Higher → more colorful.
        Returns 0.0 if the file cannot be read.
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    # Split into B, G, R channels
    B, G, R = cv2.split(img.astype("float"))
    # Compute “rg” and “yb” channels
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    # Compute the combined statistic
    std_root = np.sqrt(rg.std()**2 + yb.std()**2)
    mean_root = np.sqrt(rg.mean()**2 + yb.mean()**2)
    return std_root + 0.3 * mean_root

def get_photography_type(colorfulness: float, threshold: float) -> str:
    """
    Convert a numeric colorfulness score into a detailed photography label.
    Args:
        colorfulness: Float score from predict_colorfulness.
        threshold: Boundary between monochrome (≤) and color (>).
    Returns:
        "photography (monochrome)" or "photography (color)"
    """
    if colorfulness <= threshold:
        return "photography (monochrome)"
    else:
        return "photography (color)"

def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess an image for model inference.
    Steps:
      1. Resize to (224, 224)
      2. Scale pixel values from [0,255] to [0,1]
      3. Add batch dimension.
    Args:
        img_path: Path to an image file.
    Returns:
        A numpy array of shape (1, 224, 224, 3).
    """
    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_style(model, img_path: str) -> str:
    """
    Run the CNN model on a single image to get the base style label.
    Args:
        model: A loaded Keras model.
        img_path: Path to an image file.
    Returns:
        One of the strings in CLASS_NAMES.
    """
    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)[0]
    return CLASS_NAMES[np.argmax(preds)]

# ------------------------------------------------------------------------------
# SECTION 2: MAIN SCRIPT
# ------------------------------------------------------------------------------

def main(args):
    # 2.1 Load the dataset metadata
    print("▶ Loading HistVis metadata...")
    ds = load_dataset('csv', data_files=args.metadata_url)
    df = pd.DataFrame(ds['train'])
    print(f"✔ Loaded metadata for {len(df)} images")

    # 2.2 Download & load the style predictor model
    print("Downloading visual style predictor model...")
    model_file = hf_hub_download(
        repo_id=args.model_repo,
        filename=args.model_file
    )
    model = load_model(model_file)
    print("✔ Model loaded successfully")

    # 2.3 Iterate through images and predict styles
    results = []
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {img_path}")
            continue

        # 1) Predict the coarse style
        base_style = predict_style(model, img_path)

        # 2) If photography, refine with colorfulness
        if base_style == "photography":
            colorfulness = predict_colorfulness(img_path)
            detailed_style = get_photography_type(colorfulness, args.threshold)
        else:
            colorfulness = None
            detailed_style = base_style

        # 3) Save the result
        results.append({
            "image_path": img_path,
            "base_style": base_style,
            "detailed_style": detailed_style,
            "colorfulness": colorfulness
        })

        # Progress log every 100 images
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} images")

    # 2.4 Save all predictions to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✔ Predictions saved to: {args.output_csv}")

# ------------------------------------------------------------------------------
# SECTION 3: ARGUMENT PARSING
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict visual style of HistVis images and classify photography as monochrome or color."
    )
    parser.add_argument(
        "--metadata_url",
        default="https://huggingface.co/datasets/latentcanon/HistVis/resolve/main/dataset.csv",
        help="URL or local path to the CSV metadata file"
    )
    parser.add_argument(
        "--model_repo",
        default="mariateresadrp/visual_style_predictor",
        help="Hugging Face repository containing the Keras model"
    )
    parser.add_argument(
        "--model_file",
        default="best_vgg16_only_last.keras",
        help="Filename of the model weights within the repo"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Colorfulness threshold for monochrome vs. color photography"
    )
    parser.add_argument(
        "--output_csv",
        default="style_predictions.csv",
        help="Path where the output CSV with predictions will be saved"
    )
    args = parser.parse_args()
    main(args)
