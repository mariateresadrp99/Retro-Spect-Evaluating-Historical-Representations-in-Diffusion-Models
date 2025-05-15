#!/usr/bin/env python3
import json, time, argparse
from openai import OpenAI
from tqdm import tqdm

INSTRUCTION = (
    "Anachronism Identification: You will be provided with a list of prompts describing\n"
    "people engaged in specific activities during various historical time periods. Based on each\n"
    "prompt, identify potential anachronisms and generate a yes/no question for each:\n"
    "Answer with 'yes' if present, 'no' if absent. Return JSON with keys:\n"
    "  - possible_anachronisms: [ … ]\n"
    "  - questions_to_identify_anachronisms: { anachronism: question, … }\n"
)

def build_prompt(text):
    return [
        {"role":"system","content":INSTRUCTION},
        {"role":"user","content":f"Prompt: {text}\nReturn JSON with keys: 'possible_anachronisms' and 'questions_to_identify_anachronisms'."}
    ]

def query_llm(client, prompt):
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.0, max_tokens=512,
        messages=build_prompt(prompt)
    )
    return json.loads(resp.choices[0].message.content)

def main(input_txt, output_json):
    client = OpenAI()
    prompts = [L.strip() for L in open(input_txt) if L.strip()]
    out = []
    for idx, p in enumerate(tqdm(prompts, desc="Generating")):
        try:
            r = query_llm(client, p)
        except:
            r = {"possible_anachronisms":[], "questions_to_identify_anachronisms":{}}
        out.append({
            "index": idx,
            "prompt": p,
            "possible_anachronisms": r.get("possible_anachronisms",[]),
            "questions_to_identify_anachronisms": r.get("questions_to_identify_anachronisms",{})
        })
        time.sleep(1.0)
    json.dump(out, open(output_json,"w"), indent=2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input_txt", required=True)
    p.add_argument("--output_json", required=True)
    args=p.parse_args()
    main(args.input_txt, args.output_json)
