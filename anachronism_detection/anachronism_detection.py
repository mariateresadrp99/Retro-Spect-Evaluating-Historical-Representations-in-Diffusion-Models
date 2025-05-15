#!/usr/bin/env python3
"""
detect_anachronisms.py

Performs visual anachronism detection on HistVis images using GPT-4 Vision.

Inputs:
  --prompts_json  JSON (e.g. anachronism_detection/19th_century.json) with:
                   index, prompt, questions_to_identify_anachronisms
  --metadata_url  URL or path to CSV metadata with columns:
                   image_path, model, historical_period, universal_human_activity, category
Outputs:
  --output_json   JSON with, for each image:
                   prompt_index, prompt_text, model, image_path, image_analysis
"""

import os, json, io, argparse, base64, glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from huggingface_hub import hf_hub_download

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resize_image(image_bytes, max_size=(512,512)):
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG")
    return buf.getvalue()

def encode_image_b64(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()

def process_image(data_url, questions):
    content = [{"type":"text","text":"Identify anachronisms:\n"}]
    for k,q in questions.items():
        content.append({"type":"text","text":f"{k}: {q}\n"})
    content.append({"type":"image_url","image_url":{"url":data_url}})
    content.append({"type":"text","text":"Answer with each key: yes or no."})

    msgs = [
      {"role":"system","content":"You are an AI assistant identifying anachronisms."},
      {"role":"user","content":content}
    ]
    resp = client.chat.completions.create(
      model="gpt-4o",
      messages=msgs,
      max_tokens=500,
      temperature=0.7
    )
    ans = {}
    for line in resp.choices[0].message.content.splitlines():
        if ":" in line:
            k,v = line.split(":",1)
            ans[k.strip()] = v.strip()
    return ans

def main(prompts_json, metadata_url, output_json):
    prompts = json.load(open(prompts_json))
    df = pd.read_csv(metadata_url)
    results = []

    for item in tqdm(prompts, desc="Prompts"):
        idx    = item["index"]
        prompt = item["prompt"]
        qs     = item["questions_to_identify_anachronisms"]

        # find matching images by index
        rows = df[df.image_path.str.contains(f"/{idx}\\.")]
        for _, row in rows.iterrows():
            # extract repo & file from URL
            url = row.image_path
            repo, file = url.split("/resolve/main/")
            repo_id = repo.rsplit("/", 2)[-2] + "/" + repo.rsplit("/", 2)[-1]
            local   = hf_hub_download(repo_id=repo_id, filename=file)
            jb = open(local,"rb").read()
            jb = resize_image(jb)
            data_url = encode_image_b64(jb)
            analysis = process_image(data_url, qs)

            results.append({
                "prompt_index": idx,
                "prompt_text": prompt,
                "model": row.model,
                "image_path": url,
                "image_analysis": analysis
            })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ Saved {len(results)} detections to {output_json}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_json", required=True)
    p.add_argument("--metadata_url", required=True)
    p.add_argument("--output_json",  required=True)
    args = p.parse_args()
    main(args.prompts_json, args.metadata_url, args.output_json)
