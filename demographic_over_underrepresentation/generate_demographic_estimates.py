# generate_demographic_estimates.py
# -------------------------------------------------------------
# Uses GPT-4o to generate plausible gender & race
# distributions for each historical prompt.
# -------------------------------------------------------------
import json
import argparse
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

# Instruction string for GPT-4o
INSTRUCTION = (
    "For a given prompt [activity in time period], estimate the percentage of male vs. female, "
    "and the percentage of each major racial group (white, black, asian, indian). "
    "Base your estimates on global historical norms, social structures, and any known constraints. "
    "Output must be valid JSON with two keys: 'gender_estimate' and 'race_estimate'. "
    "Each should be a dictionary with keys 'male' and 'female' or 'white', 'black', 'asian', 'indian', "
    "whose values sum to 100."
)

def build_prompt(prompt_text):
    return [
        {
            "role": "system",
            "content": (
                "You are a historical demographics assistant. "
                "You will be given a prompt describing an activity and a historical period. "
                "Estimate the likely gender and racial breakdown in percentages, drawing from global historical contexts."
            )
        },
        {
            "role": "user",
            "content": (
                f"{INSTRUCTION}\n\n"
                f"Prompt: {prompt_text}\n\n"
                "Format example:\n"
                "{\n"
                "  \"gender_estimate\": {\"male\": 70, \"female\": 30},\n"
                "  \"race_estimate\": {\"white\": 50, \"black\": 20, \"asian\": 20, \"indian\": 10}\n"
                "}"
            )
        }
    ]

def query_llm_for_demographics(prompt_text):
    try:
        messages = build_prompt(prompt_text)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=500
        )
        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        return parsed.get("gender_estimate", {}), parsed.get("race_estimate", {})
    except Exception as e:
        print(f"❌ Error with prompt '{prompt_text}': {e}")
        return {}, {}

def main(input_file, output_file):
    # Load prompts from a JSON list, each item with a 'prompt' key
    with open(input_file, "r") as f:
        prompts = json.load(f)

    results = []
    for idx, item in enumerate(tqdm(prompts, desc="Generating demographics")):
        prompt_text = item.get("prompt", "")
        gender_est, race_est = query_llm_for_demographics(prompt_text)
        results.append({
            "index": idx,
            "prompt": prompt_text,
            "gender_estimate": gender_est,
            "race_estimate": race_est
        })
        # Optional sleep to avoid rate-limits
        time.sleep(1.0)

    # Save final results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved demographic estimates to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="JSON file with a list of prompts [{\"prompt\":\"...\"},...]")
    parser.add_argument("--output_file", required=True, help="Where to save the LLM-based demographic proposals")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
