# evaluate_styles.py

"""
Predict stylistic categories of TTI-generated images using a trained CNN classifier.
Also computes a colorfulness score to help distinguish between black-and-white and color photography.

"""

import os
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import predict_colorfulness, CLASS_NAMES


def preprocess_image(img_path):
    """
    Preprocess a single image:
    - Resizes to 224x224
    - Normalizes pixel values to [0, 1]
    - Expands dimensions for batch prediction
    """
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def predict_style(model, img_path):
    """
    Predict the stylistic category for a single image using the trained CNN.
    Returns the class label as a string.
    """
    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)
    return CLASS_NAMES[np.argmax(preds)]


def main(args):
    # Load trained CNN model
    model = load_model(args.model_path)
    print(f"✅ Loaded model from: {args.model_path}")

    results = []

    # Walk through all files in the image directory
    for root, _, files in os.walk(args.image_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            fpath = os.path.join(root, fname)

            try:
                style = predict_style(model, fpath)
                colorfulness = predict_colorfulness(fpath)

                results.append({
                    "filename": fname,
                    "relative_path": os.path.relpath(fpath, args.image_dir),
                    "style_prediction": style,
                    "colorfulness_score": colorfulness
                })
            except Exception as e:
                print(f" Skipping {fname} due to error: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"\n🎉 Done! Saved predictions for {len(df)} images to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stylistic bias in generated historical images")
    parser.add_argument("--image_dir", required=True, help="Directory containing generated images (e.g., from HistVis)")
    parser.add_argument("--model_path", default="style_classifier/best_vgg16_only_last.keras", help="Path to trained classifier weights")
    parser.add_argument("--output_file", default="style_predictions.csv", help="Path to output CSV file")
    args = parser.parse_args()
    main(args)
