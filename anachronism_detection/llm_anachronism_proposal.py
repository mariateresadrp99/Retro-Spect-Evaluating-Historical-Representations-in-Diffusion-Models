# generate_anachronism_questions_txt.py

import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import time

client = OpenAI()

INSTRUCTION = (
    "Anachronism Identification: You will be provided with a list of prompts describing "
    "people engaged in specific activities during various historical time periods. These prompts "
    "will serve as input for a Text-to-Image Generative Model like Stable Diffusion. Based on each "
    "prompt, perform the following tasks:\n"
    "• Identify potential anachronisms that might appear in the generated image. Ensure that the list "
    "is relevant to the activity, time period, and setting described in the prompt.\n"
    "• For each identified anachronism, generate a question to determine whether the anachronism appears "
    "in the generated image. Each question should end with: ‘Answer with ’yes’ (if the anachronism is present) "
    "or ’no’ (if it is absent).’\n\n"
    "All answers must be in JSON format."
)

def build_prompt(prompt_text):
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"Prompt: {prompt_text}\nPlease format your answer as JSON with keys: 'possible_anachronisms' and 'questions_to_identify_anachronisms'."}
    ]

def query_llm(prompt_text):
    try:
        messages = build_prompt(prompt_text)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=500
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error on prompt '{prompt_text}': {e}")
        return {}

def main(input_txt_path, output_json_path):
    with open(input_txt_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        response = query_llm(prompt)
        results.append({
            "index": idx,
            "prompt": prompt,
            "possible_anachronisms": response.get("possible_anachronisms", []),
            "questions_to_identify_anachronisms": response.get("questions_to_identify_anachronisms", {})
        })
        time.sleep(1.0)

    with open(output_json_path, "w") as out_file:
        json.dump(results, out_file, indent=4)
    print(f"Saved output to: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", required=True, help="Path to .txt file with one prompt per line")
    parser.add_argument("--output_json", required=True, help="Path to save the output JSON file")
    args = parser.parse_args()
    main(args.input_txt, args.output_json)
