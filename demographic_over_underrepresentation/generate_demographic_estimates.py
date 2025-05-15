#!/usr/bin/env python3
"""
generate_demographic_estimates.py

Reads a .txt file of HistVis prompts, queries GPT-4o for gender & race estimates,
and writes a CSV with one row per prompt:
  prompt, male, female, white, black, indian, middle_eastern, southeast_asian, east_asian, latino
"""

import json
import argparse
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

# initialize OpenAI client (needs OPENAI_API_KEY in env)
client = OpenAI()

INSTRUCTION = (
    "For a given prompt “[activity] in the [time period]”, estimate percentages for:\n"
    "  • gender_estimate: {\"male\": X, \"female\": Y}\n"
    "  • race_estimate: {"
    "\"white\": A, \"black\": B, \"indian\": C, "
    "\"middle_eastern\": D, \"southeast_asian\": E, "
    "\"east_asian\": F, \"latino\": G"
    "}\n"
    "Values must be integers summing to 100. Return valid JSON with those two keys."
)

def build_messages(prompt_text):
    return [
        {"role": "system", "content": "You are a historical demographics assistant."},
        {"role": "user", "content": f"{INSTRUCTION}\n\nPrompt: {prompt_text}"}
    ]

def query_llm(prompt_text):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=build_messages(prompt_text),
            temperature=0.4,
            max_tokens=256
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[Error] '{prompt_text}': {e}")
        return {"gender_estimate": {}, "race_estimate": {}}

def main(input_txt, output_csv, delay):
    prompts = [L.strip() for L in open(input_txt) if L.strip()]
    rows = []
    for prompt in tqdm(prompts, desc="Estimating demographics"):
        out = query_llm(prompt)
        ge = out.get("gender_estimate", {})
        re = out.get("race_estimate", {})
        rows.append({
            "prompt": prompt,
            "male":   ge.get("male"),
            "female": ge.get("female"),
            "white":          re.get("white"),
            "black":          re.get("black"),
            "indian":         re.get("indian"),
            "middle_eastern": re.get("middle_eastern"),
            "southeast_asian":re.get("southeast_asian"),
            "east_asian":     re.get("east_asian"),
            "latino":         re.get("latino"),
        })
        time.sleep(delay)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"→ Wrote {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV of demographic estimates")
    parser.add_argument("--input_txt",  required=True, help="One prompt per line")
    parser.add_argument("--output_csv", required=True, help="CSV output path")
    parser.add_argument("--delay",      type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()
    main(args.input_txt, args.output_csv, args.delay)
