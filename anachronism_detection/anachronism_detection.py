#!/usr/bin/env python3
"""
detect_anachronisms.py

Performs visual anachronism detection on HistVis images using GPT-4 Vision.

Takes as input:
  - A JSON file (e.g. anachronism_detection/19th_century.json) containing:
      index, prompt, and questions_to_identify_anachronisms
  - A directory tree of images named by prompt index under subfolders

Outputs:
  - A JSON file with, for each image:
      prompt_number, prompt_text, image_path, and image_analysis (question→answer)
"""

import os
import json
import base64
import glob
import io
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (requires OPENAI_API_KEY in environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resize_image(image_path, max_size=(512, 512)):
    """
    Resize image to fit within max_size, preserve aspect ratio, return JPEG bytes.
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

def encode_image_to_base64(image_bytes):
    """
    Encode raw image bytes into a base64 data URI.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def process_image_analysis(image_data_url, questions):
    """
    Send the image + questions to GPT-4 Vision and parse answers.
    Returns a dict: question_key → answer_text.
    """
    # Build content blocks
    content = [{"type": "text", "text": "Identify anachronisms in this image:\n"}]
    for key, q in questions.items():
        content.append({"type": "text", "text": f"{key}: {q}\n"})
    content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    content.append({"type": "text", "text": "Reply with each key: yes or no."})

    messages = [
        {"role": "system", "content": "You are an AI assistant identifying anachronisms."},
        {"role": "user", "content": content}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",     
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        text = resp.choices[0].message.content.strip()
        answers = {}
        for line in text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                answers[key.strip()] = val.strip()
        return answers
    except Exception as e:
        print(f"[Error] Vision API failed: {e}")
        return {}

def main(image_root, json_file, output_file):
    """
    Orchestrate detection over all prompts and images.
    """
    prompts = json.load(open(json_file))
    results = []

    for item in tqdm(prompts, desc="Prompts"):
        idx    = item["index"]
        prompt = item.get("prompt", "")
        questions = item.get("questions_to_identify_anachronisms", {})

        # Locate images under any category subfolder
        pattern = os.path.join(image_root, "*", str(idx), "*.*")
        for img_path in glob.glob(pattern):
            if not img_path.lower().endswith((".jpg",".jpeg",".png")):
                continue
            raw = resize_image(img_path)
            data_url = encode_image_to_base64(raw)
            analysis = process_image_analysis(data_url, questions)
            results.append({
                "prompt_number": idx,
                "prompt_text": prompt,
                "image_path": img_path,
                "image_analysis": analysis
            })

    # Write out detections
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ Saved detections to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect anachronisms with GPT-4 Vision")
    parser.add_argument("--image_root",  required=True, help="Base folder where images are stored")
    parser.add_argument("--json_file",   required=True, help="JSON of prompts + questions")
    parser.add_argument("--output_file", required=True, help="Output JSON path for detections")
    args = parser.parse_args()
    main(args.image_root, args.json_file, args.output_file)
