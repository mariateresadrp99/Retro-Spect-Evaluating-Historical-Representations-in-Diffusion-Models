# detect_anachronisms.py
# --------------------------------------------
# This script performs visual anachronism detection using GPT-4 Vision.
# It reads a JSON file with prompt metadata and related questions,
# encodes images into base64, sends them to the GPT-4 Turbo model along with the questions,
# and returns structured answers per image.
# --------------------------------------------

import os
import json
import base64
import glob
import io
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client using your API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Resize image to 512x512 for API compatibility while preserving aspect ratio
def resize_image(image_path, max_size=(512, 512)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

# Convert image to base64 string for API transmission
def encode_image_to_base64(image_path):
    image_data = resize_image(image_path)
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_encoded_data}"

# Send image + questions to GPT-4 Vision and extract answers
def process_image_analysis(image_data_url, questions):
    content_blocks = [{"type": "text", "text": "Analyze the image and answer these questions:\n"}]
    for key, question in questions.items():
        content_blocks.append({"type": "text", "text": f"{key}: {question}\n"})

    content_blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})
    content_blocks.append({"type": "text", "text": "Answer each question in order, starting each answer with the corresponding key."})

    messages = [
        {"role": "system", "content": "You are an AI assistant identifying anachronisms in images."},
        {"role": "user", "content": content_blocks}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        answers = {}
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                answers[key.strip()] = value.strip()
        return answers
    except Exception as e:
        print(f"Error: {e}")
        return None

# Main processing loop
def main(image_root, json_file_path, output_file):
    # Load prompt data with detection questions
    with open(json_file_path, "r") as f:
        prompt_data = json.load(f)

    results = []
    for item in tqdm(prompt_data, desc="Processing prompts"):
        prompt_number = item.get("index")
        prompt_text = item.get("prompt", "")
        questions = item.get("questions_to_identify_anachronisms", {})

        # Find subfolders like image_root/<category>/<prompt_number>
        matching_dirs = glob.glob(os.path.join(image_root, "*", str(prompt_number)))
        for image_dir in matching_dirs:
            for filename in os.listdir(image_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(image_dir, filename)
                    encoded = encode_image_to_base64(image_path)
                    analysis = process_image_analysis(encoded, questions)
                    if analysis:
                        results.append({
                            "prompt_number": prompt_number,
                            "prompt_text": prompt_text,
                            "image_path": image_path,
                            "image_analysis": analysis,
                        })

    # Save output to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f" Results saved to {output_file}")

# Entry point for CLI execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", required=True, help="Base folder for images")
    parser.add_argument("--json_file", required=True, help="Input JSON with prompts + questions")
    parser.add_argument("--output_file", required=True, help="Where to store output JSON")
    args = parser.parse_args()
    main(args.image_root, args.json_file, args.output_file)